from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import csv


@dataclass(frozen=True)
class TicketRow:
    ticket_id: str
    created_at: str  # ISO datetime
    category: str
    priority: str
    queue: str


def _daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def generate_synthetic_tickets_csv(
    out_path: Path,
    start_date: str,
    end_date: str,
    seed: int = 42,
    daily_min: int = 40,
    daily_max: int = 140,
) -> None:
    """
    Gera tickets simulados realistas para Service Desk.
    - Inclui sazonalidade (dia da semana)
    - Inclui picos aleatórios (incidentes)
    - Campos: ticket_id, created_at, category, priority, queue
    """
    random.seed(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    categories = [
        "Acesso/Conta", "VPN/Conectividade", "Email", "Hardware",
        "Software", "Impressão", "Telefonia", "Rede/Internet"
    ]
    priorities = ["P4", "P3", "P2", "P1"]  # P1 mais crítico
    queues = ["ServiceDesk-N1", "ServiceDesk-N2", "Field", "Infra"]

    rows: list[TicketRow] = []
    counter = 1

    for day in _daterange(start, end):
        dow = day.weekday()  # 0=Mon..6=Sun
        # padrão típico: seg/ter alto, fim de semana baixo
        dow_factor = {
            0: 1.20,  # segunda
            1: 1.10,
            2: 1.00,
            3: 0.95,
            4: 0.90,  # sexta
            5: 0.55,  # sábado
            6: 0.50,  # domingo
        }[dow]

        base = random.randint(daily_min, daily_max)
        n = int(base * dow_factor)

        # picos (tipo incidente): ~3% dos dias
        if random.random() < 0.03:
            n = int(n * random.uniform(1.6, 2.4))

        for _ in range(n):
            # horários concentrados 08h-18h (com ruído)
            hour = int(min(23, max(0, random.gauss(mu=13, sigma=3))))
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            created = datetime(day.year, day.month, day.day, hour, minute, second).isoformat(sep=" ")

            cat = random.choices(
                categories,
                weights=[18, 14, 12, 10, 16, 8, 6, 16],
                k=1
            )[0]

            pr = random.choices(
                priorities,
                weights=[55, 30, 12, 3],  # maioria P4/P3
                k=1
            )[0]

            q = random.choices(
                queues,
                weights=[60, 22, 8, 10],
                k=1
            )[0]

            ticket_id = f"TCK-{day.strftime('%Y%m%d')}-{counter:06d}"
            counter += 1
            rows.append(TicketRow(ticket_id, created, cat, pr, q))

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticket_id", "created_at", "category", "priority", "queue"])
        for r in rows:
            writer.writerow([r.ticket_id, r.created_at, r.category, r.priority, r.queue])