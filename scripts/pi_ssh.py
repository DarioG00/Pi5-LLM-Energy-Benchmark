"""Connessione SSH al Raspberry Pi 5 e pilotaggio interattivo di llama-cli.

Usa Paramiko con una shell interattiva (invoke_shell): il modello viene avviato
una sola volta per configurazione e resta caricato in memoria; i prompt dei
benchmark vengono inviati uno dopo l'altro attendendo ogni volta il ritorno del
prompt `>` di llama.cpp.
"""
from __future__ import annotations

import re
import time
import base64
import logging
from dataclasses import dataclass
from typing import Optional

import paramiko

log = logging.getLogger("ssh")

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# Riga dei timing di fine generazione: "[ Prompt: X t/s | Generation: Y t/s ]"
_PERF_RE = re.compile(r"Prompt:\s*[\d.,]+\s*t/s\s*\|\s*Generation:\s*[\d.,]+\s*t/s", re.IGNORECASE)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "")


@dataclass
class PiConfig:
    host: str = "192.168.10.2"
    username: str = "dario"
    password: str = ""
    ssh_port: int = 22
    boot_wait_seconds: int = 90
    boot_retry_seconds: int = 5

    @classmethod
    def from_dict(cls, d: dict) -> "PiConfig":
        known = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(**known)


class PiSSH:
    def __init__(self, cfg: PiConfig, ready_prompt: str = "> ",
                 prompt_mode: str = "read",
                 prompt_file: str = "/tmp/bench_prompt.txt"):
        self.cfg = cfg
        self.ready_prompt = ready_prompt
        # "inline" (default consigliato): invia il prompt come UNA riga (a-capo
        #          sostituiti da spazi). Affidabile su tutti i build di llama-cli.
        # "read":  SPERIMENTALE. Usa il comando /read di llama-cli per leggere il
        #          prompt da file preservando gli a-capo, ma su alcuni build /read
        #          allega il file come contesto invece di inviarlo come messaggio:
        #          in quel caso la generazione non parte. Usare solo se verificato.
        self.prompt_mode = prompt_mode
        self.prompt_file = prompt_file
        self.client: Optional[paramiko.SSHClient] = None
        self.shell: Optional[paramiko.Channel] = None
        self._staged: Optional[str] = None

    # ------------------------------------------------------------- connessione
    def wait_for_boot_and_connect(self) -> None:
        """Tenta la connessione SSH finché il Pi non ha terminato il boot."""
        deadline = time.time() + self.cfg.boot_wait_seconds
        last_err: Optional[Exception] = None
        log.info("Attesa boot del Raspberry Pi e connessione SSH a %s...", self.cfg.host)
        while time.time() < deadline:
            try:
                self._open()
                log.info("Connessione SSH stabilita con %s", self.cfg.host)
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(self.cfg.boot_retry_seconds)
        raise TimeoutError(
            f"Impossibile connettersi a {self.cfg.host} entro "
            f"{self.cfg.boot_wait_seconds}s: {last_err}"
        )

    def _open(self) -> None:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=self.cfg.host,
            port=self.cfg.ssh_port,
            username=self.cfg.username,
            password=self.cfg.password,
            timeout=10,
            banner_timeout=10,
            auth_timeout=10,
        )
        self.client = client

    # ----------------------------------------------------------------- shell
    def open_shell(self) -> None:
        assert self.client is not None, "SSH non connesso"
        self.shell = self.client.invoke_shell(width=200, height=50)
        self.shell.settimeout(0.5)
        time.sleep(1.0)
        self._drain()  # svuota il banner di login

    def _drain(self) -> str:
        """Legge tutto ciò che è disponibile senza bloccare."""
        out = ""
        if self.shell is None:
            return out
        try:
            while self.shell.recv_ready():
                out += self.shell.recv(65536).decode("utf-8", "ignore")
        except Exception:
            pass
        return out

    def _read_until_ready(self, timeout: float, expect_perf: bool = False) -> str:
        """Accumula output fino al segnale di completamento o allo scadere del timeout.

        Con ``expect_perf=True`` (invio di un prompt) il completamento e' segnalato
        dalla riga dei timing ``[ Prompt: X t/s | Generation: Y t/s ]``, che e'
        inequivocabile e non risente dei caratteri ``>`` presenti nel testo del
        prompt o della risposta (es. ``->`` o ``>>>``). Con ``expect_perf=False``
        (attesa dopo il caricamento del modello) si attende invece la riga di prompt
        pronta, costituita dal solo carattere ``>``.
        """
        assert self.shell is not None
        buf = ""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                chunk = self.shell.recv(65536).decode("utf-8", "ignore")
                if chunk:
                    buf += chunk
                    clean = strip_ansi(buf)
                    if expect_perf:
                        # fine generazione: attende SOLO la riga dei timing, cosi'
                        # l'eco iniziale "> " e i caratteri '>' del testo non
                        # interrompono la lettura in anticipo.
                        if _PERF_RE.search(clean):
                            return clean
                    else:
                        # attesa del prompt pronto dopo il caricamento del modello
                        tail = clean.rstrip("\n ").splitlines()
                        if tail and tail[-1].strip() == ">":
                            return clean
                else:
                    time.sleep(0.05)
            except paramiko.ssh_exception.SSHException:
                time.sleep(0.05)
            except Exception:
                time.sleep(0.05)
        log.warning("Timeout in attesa del prompt '>' dopo %.0fs", timeout)
        return strip_ansi(buf)

    # ----------------------------------------------------- comandi paralleli
    def run_command(self, cmd: str, timeout: float = 15.0) -> str:
        """Esegue un comando one-shot via canale SSH separato (exec_command).

        Usato per `vcgencmd` mentre la shell interattiva tiene llama-cli aperto:
        i due canali sono indipendenti sulla stessa connessione SSH.
        """
        assert self.client is not None, "SSH non connesso"
        stdin, stdout, stderr = self.client.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", "ignore")
        return out.strip()

    def measure_temp(self) -> float:
        """Temperatura della SoC in °C (vcgencmd measure_temp)."""
        try:
            raw = self.run_command("vcgencmd measure_temp")
            m = re.search(r"temp=([\d.]+)", raw)
            return float(m.group(1)) if m else float("nan")
        except Exception as exc:
            log.warning("measure_temp fallita: %s", exc)
            return float("nan")

    def get_throttled_raw(self) -> str:
        """Valore grezzo di `vcgencmd get_throttled` (es. 'throttled=0x0')."""
        try:
            return self.run_command("vcgencmd get_throttled")
        except Exception as exc:
            log.warning("get_throttled fallita: %s", exc)
            return ""

    # ------------------------------------------------------------ llama-cli
    def clear_history(self) -> None:
        """Ripulisce la cronologia della chat di llama-cli (comando \texttt{/clear}).

        Va chiamato prima di ogni prompt: cosi' ogni inferenza e' valutata in modo
        indipendente e il contesto non si accumula fino a saturare la finestra
        impostata con ``-c`` (evita l'errore "exceeds the available context size").
        """
        if self.shell is None:
            return
        try:
            self._drain()
            self.shell.send("/clear\n")
            self._read_until_ready(15)
            self._drain()
        except Exception as exc:  # noqa: BLE001
            log.warning("clear_history fallita: %s", exc)

    def launch_model(self, command: str, load_timeout: float) -> str:
        """Avvia llama-cli in modalità interattiva e attende che sia pronto."""
        assert self.shell is not None, "shell non aperta"
        self._drain()
        log.info("Avvio modello: %s", command)
        self.shell.send(command + "\n")
        out = self._read_until_ready(load_timeout)
        return out

    def stage_prompt(self, prompt: str) -> None:
        """Prepara il prompt SENZA avviare la generazione.

        Va chiamato PRIMA di aprire la finestra di misura, cosi' l'overhead di
        preparazione (scrittura file + \texttt{/read}) non finisce nell'energia e
        nella latenza dell'inferenza. In modalita' "read" scrive il prompt in un
        file sul Pi (a-capo preservati) e lo carica nel buffer con \texttt{/read};
        in modalita' "inline" memorizza il prompt collassato su una riga.
        """
        assert self.shell is not None
        self._drain()
        if self.prompt_mode == "read":
            b64 = base64.b64encode(prompt.encode("utf-8")).decode("ascii")
            self.run_command(f"echo {b64} | base64 -d > {self.prompt_file}")
            self.shell.send(f"/read {self.prompt_file}\n")
            time.sleep(0.4)          # attende il caricamento del file nel buffer
            self._drain()            # scarta la conferma/eco del comando /read
            self._staged = None
        else:
            self._staged = prompt.replace("\n", " ").strip()

    def submit_prompt(self, infer_timeout: float) -> str:
        """Avvia la generazione del prompt gia' preparato e attende il completamento
        (rilevato dalla riga dei timing). Ritorna l'output grezzo."""
        assert self.shell is not None
        if self.prompt_mode == "read":
            self.shell.send("\n")               # invia il contenuto caricato
        else:
            self.shell.send((self._staged or "") + "\n")
        return self._read_until_ready(infer_timeout, expect_perf=True)

    def send_prompt(self, prompt: str, infer_timeout: float) -> str:
        """Prepara e invia un prompt in un solo passo (staging + submit)."""
        self.stage_prompt(prompt)
        return self.submit_prompt(infer_timeout)

    def stop_model(self) -> str:
        """Esce da llama-cli (Ctrl-C) e raccoglie il riepilogo finale dei timing."""
        if self.shell is None:
            return ""
        try:
            self.shell.send("\x03")  # Ctrl-C
            time.sleep(1.5)
            out = self._drain()
            return strip_ansi(out)
        except Exception as exc:
            log.warning("Errore nello stop del modello: %s", exc)
            return ""

    # ----------------------------------------------------------------- close
    def close_shell(self) -> None:
        try:
            if self.shell is not None:
                self.shell.close()
                self.shell = None
        except Exception:
            pass

    def close(self) -> None:
        self.close_shell()
        try:
            if self.client is not None:
                self.client.close()
                self.client = None
                log.info("Connessione SSH chiusa.")
        except Exception:
            pass
