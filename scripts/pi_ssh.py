"""Connessione SSH al Raspberry Pi 5 e pilotaggio interattivo di llama-cli.

Usa Paramiko con una shell interattiva (invoke_shell): il modello viene avviato
una sola volta per configurazione e resta caricato in memoria; i prompt dei
benchmark vengono inviati uno dopo l'altro attendendo ogni volta il ritorno del
prompt `>` di llama.cpp.
"""
from __future__ import annotations

import re
import time
import logging
from dataclasses import dataclass
from typing import Optional

import paramiko

log = logging.getLogger("ssh")

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


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
    def __init__(self, cfg: PiConfig, ready_prompt: str = "> "):
        self.cfg = cfg
        self.ready_prompt = ready_prompt
        self.client: Optional[paramiko.SSHClient] = None
        self.shell: Optional[paramiko.Channel] = None

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

    def _read_until_ready(self, timeout: float) -> str:
        """Accumula output finché ricompare il prompt `>` o scade il timeout."""
        assert self.shell is not None
        buf = ""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                chunk = self.shell.recv(65536).decode("utf-8", "ignore")
                if chunk:
                    buf += chunk
                    clean = strip_ansi(buf)
                    # il prompt di llama.cpp compare a inizio riga in attesa di input
                    tail = clean.rstrip("\n ").splitlines()
                    if tail and tail[-1].strip().endswith(">"):
                        return clean
                    if clean.rstrip().endswith(">"):
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
    def launch_model(self, command: str, load_timeout: float) -> str:
        """Avvia llama-cli in modalità interattiva e attende che sia pronto."""
        assert self.shell is not None, "shell non aperta"
        self._drain()
        log.info("Avvio modello: %s", command)
        self.shell.send(command + "\n")
        out = self._read_until_ready(load_timeout)
        return out

    def send_prompt(self, prompt: str, infer_timeout: float) -> str:
        """Invia un prompt e ritorna il testo generato fino al prompt `>`."""
        assert self.shell is not None
        self._drain()
        # i prompt possono essere multi-riga: sostituiamo i newline interni con
        # spazi per evitare invii prematuri, mantenendo la struttura leggibile.
        single = prompt.replace("\n", " ").strip()
        self.shell.send(single + "\n")
        out = self._read_until_ready(infer_timeout)
        return out

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
