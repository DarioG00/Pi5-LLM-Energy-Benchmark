"""Misura del consumo del Raspberry Pi 5 tramite il PMIC (backend PMIC).

Riprende il metodo del progetto **jfikar/RPi5-power**
(https://github.com/jfikar/RPi5-power): il PMIC del Pi 5 espone tensione e
corrente di 12 rami; la potenza istantanea e' la somma dei prodotti I*V sui 12
rami, letta con `vcgencmd pmic_read_adc`.

Per impostazione predefinita si registra direttamente questa somma (nessuna
correzione): e' una misura coerente e ripetibile del consumo interno della
scheda, adatta al confronto tra modelli e quantizzazioni. La correzione lineare
proposta da jfikar (slope 1.1451, offset 0.5879) per stimare il consumo reale
alla presa e' specifica della sua scheda+alimentatore e va usata solo se
ricalibrata sul proprio setup (parametri corr_slope/corr_offset).

Riferimento (script originale, 1 Hz):
        https://github.com/jfikar/RPi5-power/blob/main/rpi5_power.sh

Il campionamento avviene in un thread in background via SSH, con timestamp sul
clock monotono dell'host (lo stesso usato per delimitare le finestre di
inferenza): non serve quindi sincronizzare gli orologi di host e Pi. La classe
`PmicMonitor` espone la stessa interfaccia usata dal `BenchmarkRunner`.

Robustezza: ogni lettura chiude esplicitamente la propria sessione SSH (per non
esaurire i canali del server su campagne lunghe) e, se le letture cominciano a
fallire, il campionatore si riconnette e riprende automaticamente. Se una
registrazione dovesse restare senza campioni, viene emesso un WARNING (cosi' il
problema e' visibile e non passa inosservato come un'energia pari a zero).

Il Pi 5 e' alimentato dall'alimentatore ufficiale (5,1 V / 3 A): il PMIC misura
il consumo senza alcun hardware esterno, evitando i rischi legati
all'alimentazione via GPIO.
"""
from __future__ import annotations

import re
import time
import logging
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

# paramiko viene importato in modo lazy dentro connect(): cosi' il parser e la
# simulazione restano utilizzabili anche dove paramiko non e' installato.

log = logging.getLogger("pmic")

# Correzione lineare opzionale delle letture del PMIC.
#
# Per impostazione predefinita NON si applica alcuna correzione (slope=1,
# offset=0): si registra quindi la potenza misurata direttamente dal PMIC come
# somma I*V sui 12 rami. Questa e' una misura coerente e ripetibile del consumo
# interno della scheda, adatta al confronto tra modelli/quantizzazioni.
#
# I coefficienti (slope=1.1451, offset=0.5879) proposti dal progetto
# jfikar/RPi5-power per stimare il consumo reale alla presa sono stati ricavati
# su UNA specifica scheda+alimentatore e non sono trasferibili ad altre unita':
# vanno quindi usati solo se ricalibrati sul proprio setup.
CORR_SLOPE = 1.0
CORR_OFFSET = 0.0

# Correnti (..._A current(n)=...A) e tensioni (..._V volt(n)=...V) dei rami.
_CUR_RE = re.compile(r"_A\s+current\(\d+\)=([-\d.]+)A")
_VOLT_RE = re.compile(r"_V\s+volt\(\d+\)=([-\d.]+)V")

PMIC_CMD = "vcgencmd pmic_read_adc"


def parse_pmic_power(raw: str, slope: float = CORR_SLOPE,
                     offset: float = CORR_OFFSET) -> Optional[float]:
    """Potenza (W) da un output di ``vcgencmd pmic_read_adc``.

    Somma i prodotti I*V dei rami (le correnti abbinate, nell'ordine di
    apparizione, alle corrispondenti tensioni). Con slope=1/offset=0 (default)
    restituisce la somma grezza; slope/offset diversi applicano una correzione
    lineare. Ritorna ``None`` se l'output non e' interpretabile.
    """
    currents = [float(x) for x in _CUR_RE.findall(raw)]
    volts = [float(x) for x in _VOLT_RE.findall(raw)]
    if not currents or not volts:
        return None
    n = min(len(currents), len(volts))
    p_pmic = sum(currents[i] * volts[i] for i in range(n))
    return p_pmic * slope + offset


@dataclass
class PmicConfig:
    """Parametri di campionamento del PMIC (frequenza, durata idle, assestamento, coefficienti di correzione)."""
    sample_hz: float = 10.0          # cadenza di campionamento richiesta (Hz)
    idle_seconds: int = 20           # durata misura idle (bias)
    settle_seconds: int = 3          # assestamento prima di ogni registrazione
    corr_slope: float = CORR_SLOPE
    corr_offset: float = CORR_OFFSET

    @classmethod
    def from_dict(cls, d: dict) -> "PmicConfig":
        """Crea una PmicConfig da un dizionario, ignorando le chiavi non pertinenti."""
        known = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(**known)


class PmicMonitor:
    """Campionatore del PMIC (stessa interfaccia usata dal BenchmarkRunner).

    Apre una connessione SSH **dedicata** al Pi (indipendente da quella
    interattiva usata per llama-cli) e legge il PMIC in un thread in background.
    """

    # dopo quante letture consecutive fallite si tenta la riconnessione
    _FAIL_BEFORE_RECONNECT = 3
    # ogni quante letture fallite si ritenta la riconnessione (a ~10 Hz ~2 s)
    _RECONNECT_EVERY = 20

    def __init__(self, pi_cfg: Any, cfg: PmicConfig):
        """Inizializza il monitor con la configurazione del Pi e i parametri di campionamento del PMIC."""
        self.pi_cfg = pi_cfg          # PiConfig (host/username/password/ssh_port)
        self.cfg = cfg
        self.settle_seconds = cfg.settle_seconds
        self._client: Optional[Any] = None   # paramiko.SSHClient
        self.idle_power_w: float = 0.0
        self._rec_t0: float = 0.0
        # campioni della registrazione corrente: lista di (t_relativo_s, potenza_W)
        self._samples: List[Tuple[float, float]] = []
        self._sampling = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ setup
    def _open_client(self):
        """Apre e restituisce una connessione SSH dedicata alla lettura del PMIC."""
        import paramiko
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=self.pi_cfg.host,
            port=self.pi_cfg.ssh_port,
            username=self.pi_cfg.username,
            password=self.pi_cfg.password,
            timeout=10, banner_timeout=10, auth_timeout=10,
        )
        return client

    def connect(self) -> None:
        """Apre la connessione SSH dedicata al campionamento del PMIC."""
        log.info("Connessione SSH (PMIC) a %s...", self.pi_cfg.host)
        self._client = self._open_client()
        p = self._read_power()
        if p is None:
            log.warning("Il PMIC non ha restituito un output valido al primo "
                        "tentativo: verificare 'vcgencmd pmic_read_adc' sul Pi.")
        else:
            log.info("PMIC pronto: potenza istantanea iniziale %.3f W", p)

    def _reconnect(self) -> bool:
        """Chiude e riapre la connessione SSH del PMIC. True se riuscita."""
        try:
            if self._client is not None:
                self._client.close()
        except Exception:  # noqa: BLE001
            pass
        self._client = None
        try:
            self._client = self._open_client()
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("Riconnessione PMIC fallita: %s", exc)
            return False

    def configure_power(self) -> None:
        """Nessuna configurazione: il Pi e' alimentato dall'alimentatore ufficiale."""
        log.info("Backend PMIC: alimentazione dall'alimentatore ufficiale del Pi.")

    def power_on(self) -> None:
        """Il Pi e' gia' acceso: nulla da fare (compatibilita' con l'interfaccia)."""
        log.info("Backend PMIC: il Raspberry Pi e' alimentato esternamente e gia' acceso.")

    def power_off(self) -> None:
        """Non spegniamo il Pi (alimentato esternamente)."""
        pass

    # --------------------------------------------------------------- lettura
    def _read_power(self) -> Optional[float]:
        """Una lettura del PMIC -> potenza (W). Chiude sempre la sessione SSH."""
        if self._client is None:
            return None
        out = None
        try:
            _in, out, _err = self._client.exec_command(PMIC_CMD, timeout=8)
            raw = out.read().decode("utf-8", "ignore")
        except Exception as exc:  # noqa: BLE001
            log.debug("Lettura PMIC fallita: %s", exc)
            return None
        finally:
            # chiude esplicitamente il canale: evita di accumulare sessioni SSH
            # (che su campagne lunghe esaurirebbero i canali del server).
            try:
                if out is not None:
                    out.channel.close()
            except Exception:  # noqa: BLE001
                pass
        return parse_pmic_power(raw, self.cfg.corr_slope, self.cfg.corr_offset)

    def _sampler_loop(self) -> None:
        """Ciclo del thread in background: legge la potenza dal PMIC alla frequenza impostata e, se le letture falliscono, tenta la riconnessione."""
        period = 1.0 / max(0.5, self.cfg.sample_hz)
        fails = 0
        warned = False
        while self._sampling:
            t = self.mark()
            p = self._read_power()
            if p is not None:
                if fails and warned:
                    log.info("Campionamento PMIC ripristinato.")
                fails = 0
                warned = False
                with self._lock:
                    self._samples.append((t, p))
            else:
                fails += 1
                if fails == self._FAIL_BEFORE_RECONNECT and not warned:
                    log.warning("Campionamento PMIC: letture non valide, "
                                "tento la riconnessione...")
                    warned = True
                if (fails >= self._FAIL_BEFORE_RECONNECT
                        and (fails - self._FAIL_BEFORE_RECONNECT) % self._RECONNECT_EVERY == 0):
                    if self._sampling:
                        self._reconnect()
            dt = self.mark() - t
            time.sleep(max(0.0, period - dt))

    # --------------------------------------------------------------- progetto
    def new_project(self):
        """Compatibilita' con l'interfaccia: nessun progetto da creare."""
        return None

    # ------------------------------------------------------------ recording
    def start_recording(self) -> None:
        """Avvia una nuova registrazione e il thread di campionamento in background."""
        with self._lock:
            self._samples = []
        self._rec_t0 = time.monotonic()
        self._sampling = True
        self._thread = threading.Thread(target=self._sampler_loop, daemon=True)
        self._thread.start()
        log.debug("Campionamento PMIC avviato (%.0f Hz).", self.cfg.sample_hz)

    def stop_recording(self):
        """Ferma il campionamento e restituisce i campioni raccolti (avvisa se la registrazione e' vuota)."""
        self._sampling = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        with self._lock:
            samples = list(self._samples)
        if not samples:
            log.warning("Registrazione PMIC senza campioni: la connessione al Pi "
                        "potrebbe essere caduta. Energia non misurata per questa run.")
        else:
            log.debug("Campionamento PMIC fermato (%d campioni).", len(samples))
        return samples

    def mark(self) -> float:
        """Tempo (s) relativo all'inizio della registrazione."""
        return time.monotonic() - self._rec_t0

    # --------------------------------------------------------------- misure
    def measure_idle_bias(self) -> float:
        """Registra il consumo idle del Pi e ne ricava la potenza media (W)."""
        log.info("Misura del consumo idle (PMIC) per %d s...", self.cfg.idle_seconds)
        self.start_recording()
        time.sleep(self.cfg.idle_seconds)
        samples = self.stop_recording()
        powers = [p for _, p in samples]
        self.idle_power_w = sum(powers) / len(powers) if powers else 0.0
        log.info("Potenza idle media (PMIC): %.4f W (%d campioni)",
                 self.idle_power_w, len(powers))
        return self.idle_power_w

    def window_energy(self, rec, t_from: float, t_to: float) -> dict:
        """Energia e potenza in [t_from, t_to] (metodo dei trapezi sui campioni)."""
        t_from = max(0.0, t_from)
        if t_to <= t_from:
            t_to = t_from + 1e-3
        duration = t_to - t_from
        pts = [(t, p) for (t, p) in (rec or []) if t_from <= t <= t_to]

        no_data = False
        if len(pts) >= 2:
            energy_trap = 0.0
            for (ta, pa), (tb, pb) in zip(pts, pts[1:]):
                energy_trap += 0.5 * (pa + pb) * (tb - ta)
            span = pts[-1][0] - pts[0][0]
            avg_power = energy_trap / span if span > 0 else pts[0][1]
            energy_total = avg_power * duration
        elif len(pts) == 1:
            avg_power = pts[0][1]
            energy_total = avg_power * duration
        else:
            avg_power = self._nearest_power(rec, 0.5 * (t_from + t_to))
            energy_total = avg_power * duration
            no_data = not rec  # nessun campione nell'intera registrazione

        energy_idle = self.idle_power_w * duration
        energy_net = energy_total - energy_idle
        return {
            "duration_s": duration,
            "avg_power_w": avg_power,
            "energy_total_j": energy_total,
            "energy_idle_j": energy_idle,
            "energy_net_j": energy_net,
            "no_data": no_data,
        }

    @staticmethod
    def _nearest_power(rec, t: float) -> float:
        """Restituisce la potenza del campione temporalmente piu' vicino all'istante indicato."""
        if not rec:
            return 0.0
        return min(rec, key=lambda s: abs(s[0] - t))[1]

    # ----------------------------------------------------------------- close
    def disconnect(self) -> None:
        """Ferma il campionamento e chiude la connessione SSH del PMIC."""
        self._sampling = False
        try:
            if self._thread is not None:
                self._thread.join(timeout=2.0)
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._client is not None:
                self._client.close()
                self._client = None
                log.info("Connessione SSH (PMIC) chiusa.")
        except Exception as exc:  # noqa: BLE001
            log.warning("Errore in disconnessione PMIC: %s", exc)
