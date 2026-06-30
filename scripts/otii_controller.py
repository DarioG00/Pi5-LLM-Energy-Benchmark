"""Controllo dell'Otii Arc tramite l'API TCP (otii_tcp_client).

Gestisce: connessione al TCP server, configurazione dell'alimentazione
(5 V, limite di corrente), abilitazione dei canali di misura, accensione del
Raspberry Pi 5, registrazione del consumo idle (usato come bias) e calcolo
dell'energia consumata in una finestra temporale.

Riferimento API: https://www.qoitech.com/help/tcpserver/index.html
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from otii_tcp_client import otii_client

log = logging.getLogger("otii")


@dataclass
class OtiiConfig:
    host: str = "127.0.0.1"
    port: int = 1905
    main_voltage: float = 5.0
    supply_current: float = 2.4
    max_current: float = 5.0
    channels: tuple = ("mc", "mv", "mp")
    power_channel: str = "mp"
    samplerate: int = 4000
    project_dir: str = "./recordings"
    idle_seconds: int = 20
    settle_seconds: int = 3

    @classmethod
    def from_dict(cls, d: dict) -> "OtiiConfig":
        known = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        if "channels" in known:
            known["channels"] = tuple(known["channels"])
        return cls(**known)


class OtiiController:
    """Wrapper di alto livello sull'Otii Arc."""

    def __init__(self, cfg: OtiiConfig):
        self.cfg = cfg
        self._client: Optional[otii_client.OtiiClient] = None
        self.otii = None
        self.device = None          # arc.Arc
        self.project = None
        self.idle_power_w: float = 0.0   # bias medio in Watt
        self._rec_t0: float = 0.0        # monotonic all'avvio della registrazione

    # ------------------------------------------------------------------ setup
    def connect(self) -> None:
        """Apre la connessione al TCP server e seleziona il dispositivo Arc."""
        log.info("Connessione al TCP server Otii %s:%s", self.cfg.host, self.cfg.port)
        self._client = otii_client.OtiiClient()
        self.otii = self._client.connect(host=self.cfg.host, port=self.cfg.port)
        devices = self.otii.get_devices()
        if not devices:
            raise RuntimeError("Nessun dispositivo Otii Arc trovato dal TCP server.")
        self.device = devices[0]
        log.info("Dispositivo Otii selezionato: %s", getattr(self.device, "name", self.device.id))

    def configure_power(self) -> None:
        """Imposta 5 V, limite di corrente e abilita i canali di misura.

        Il Pi 5 può avere picchi fino a ~5 A in carico: il limite (max_current)
        viene quindi alzato a 5 A per non interrompere l'alimentazione, mentre il
        valore nominale di corrente del generatore resta a 2.4 A.
        """
        dev = self.device
        dev.set_main_voltage(self.cfg.main_voltage)
        # set_main_current ha effetto solo in modalità constant-current; lo
        # impostiamo comunque come riferimento del generatore.
        try:
            dev.set_main_current(self.cfg.supply_current)
        except Exception:  # non tutte le firmware espongono il comando in CV
            log.debug("set_main_current non applicabile in questa modalità")
        dev.set_max_current(self.cfg.max_current)
        for ch in self.cfg.channels:
            dev.enable_channel(ch, True)
            try:
                dev.set_channel_samplerate(ch, self.cfg.samplerate)
            except Exception:
                pass
        log.info("Alimentazione configurata: %.1f V, limite %.1f A, canali %s",
                 self.cfg.main_voltage, self.cfg.max_current, list(self.cfg.channels))

    def power_on(self) -> None:
        """Accende l'uscita: il Raspberry Pi inizia il boot."""
        self.device.set_main(True)
        log.info("Uscita di alimentazione ON: il Raspberry Pi sta avviandosi.")

    def power_off(self) -> None:
        try:
            self.device.set_main(False)
            log.info("Uscita di alimentazione OFF.")
        except Exception as exc:
            log.warning("Impossibile spegnere l'uscita: %s", exc)

    # --------------------------------------------------------------- progetto
    def new_project(self):
        self.project = self.otii.create_project()
        return self.project

    # ------------------------------------------------------------ recording
    def start_recording(self) -> None:
        self.project.start_recording()
        self._rec_t0 = time.monotonic()
        log.debug("Registrazione avviata.")

    def stop_recording(self):
        self.project.stop_recording()
        log.debug("Registrazione fermata.")
        return self.project.get_last_recording()

    def mark(self) -> float:
        """Restituisce il tempo (s) relativo all'inizio della registrazione."""
        return time.monotonic() - self._rec_t0

    # --------------------------------------------------------------- misure
    def measure_idle_bias(self) -> float:
        """Registra il consumo idle del Pi e ne ricava la potenza media (W).

        Va chiamato a Pi acceso ma a riposo. Il valore viene usato come bias da
        sottrarre all'energia di ogni inferenza.
        """
        log.info("Misura del consumo idle per %d s...", self.cfg.idle_seconds)
        self.new_project()
        self.start_recording()
        time.sleep(self.cfg.idle_seconds)
        rec = self.stop_recording()
        stats = rec.get_channel_statistics(
            self.device.id, self.cfg.power_channel, 0.0, self.cfg.idle_seconds
        )
        self.idle_power_w = float(stats.get("average", 0.0))
        log.info("Potenza idle media: %.4f W", self.idle_power_w)
        return self.idle_power_w

    def window_energy(self, rec, t_from: float, t_to: float) -> dict:
        """Energia e potenza in [t_from, t_to] (s, relativi alla registrazione).

        Ritorna un dizionario con energia totale (J), energia idle stimata (J),
        energia netta dell'inferenza (J), potenza media (W) e durata (s).
        """
        t_from = max(0.0, t_from)
        if t_to <= t_from:
            t_to = t_from + 1e-3
        stats = rec.get_channel_statistics(
            self.device.id, self.cfg.power_channel, t_from, t_to
        )
        duration = t_to - t_from
        energy_total = float(stats.get("energy", 0.0))
        avg_power = float(stats.get("average", 0.0))
        energy_idle = self.idle_power_w * duration
        energy_net = energy_total - energy_idle
        return {
            "duration_s": duration,
            "avg_power_w": avg_power,
            "energy_total_j": energy_total,
            "energy_idle_j": energy_idle,
            "energy_net_j": energy_net,
        }

    # ----------------------------------------------------------------- close
    def disconnect(self) -> None:
        try:
            if self._client is not None:
                self._client.disconnect()
                log.info("Disconnesso dal TCP server Otii.")
        except Exception as exc:
            log.warning("Errore in disconnessione Otii: %s", exc)
