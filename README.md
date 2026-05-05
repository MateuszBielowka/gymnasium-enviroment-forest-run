# Gymnasium Enviroments - Starter Project

Ten projekt zawiera:
- wlasne srodowisko `GridWorldEnv` dla Gymnasium,
- gotowe wrappery,
- przykladowe skrypty,
- testy

W aktualnej wersji gry agent startuje w lewym dolnym rogu, a celem jest prawy gorny rog. Na mapie losowo pojawiaja sie nieprzechodnie drzewa oraz krzewy, ktore zabieraja punkty przy wejściu.

## 1. Szybki start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Uruchom losowa gre:

```bash
python scripts/run_gridworld.py
```

Uruchom prosty trening Q-learning:

```bash
python scripts/train_qlearning.py
```

Uruchom testy:

```bash
pip install -e .[dev]
pytest
```

