# Asterion 5G + ASTERIX

Інтегрований проєкт для:

- симуляції RF-вимірювань 5G gNodeB;
- реконструкції траєкторій і швидкості об'єкта;
- кодування треків у ASTERIX CAT 062;
- збереження симульованих та реальних даних у форматі Apache Parquet;
- навчання TensorFlow-моделі часу руління літака;
- запуску всього процесу через Bash та графічний інтерфейс CustomTkinter.

## Архітектура

```

The launcher keeps GPU acceleration disabled by default for the desktop
pipeline. This avoids CUDA runtime conflicts between TensorFlow, PyTorch, and
Sionna on systems without a matching CUDA installation. The model still runs
on the CPU.text
5Gsim/
  gNodeB.py              RF-симулятор, канал MIMO та DSP
  rf_sensing.py          API траєкторій і TrackEstimate
  router.py              Передача треків між етапами
  demo.py                Безперервний 5G-демо-запуск

integration/
  bridge.py              5G TrackEstimate -> ASTERIX CAT 062 -> Parquet

yezhik-asterix/
  asterix_wrapper.py     ASTERIX CAT 062 encoder
  run_pipeline.py        Обробка Parquet і генерація симуляційних точок

Model/
  data_loader.py         Завантаження та очищення Parquet
  feature_engineering.py Підготовка ознак
  train_model.py         Навчання TensorFlow-моделі
  inference.py           Генерація прогнозів
  evaluator.py           Оцінювання моделі

app.py                   CustomTkinter GUI
asterion_5g.sh           Unified launcher for all Bash entrypoints
run_app.sh               Запуск GUI
run_5g.sh                Запуск 5G-демо
run_model.sh             Train/predict/evaluate через Bash
```

## Вимоги

- Linux/macOS з Bash;
- Python 3.10+;
- віртуальне середовище `.venv`;
- залежності з `yezhik-asterix/requirements.txt`;
- для GUI потрібен графічний сеанс (X11/Wayland).

## Встановлення

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r yezhik-asterix/requirements.txt
```

Якщо Python знаходиться не у `.venv/bin/python`, можна передати власний інтерпретатор:

```bash
PYTHON=/path/to/python ./run_app.sh
```

## Графічний інтерфейс

Запуск:

```bash
./run_app.sh
```

The unified launcher can be used instead:

```bash
./asterion_5g.sh gui
```

GUI дозволяє:

1. Вказати кількість RF-зразків.
2. Запустити симуляцію 5G.
3. Перетворити результати у CAT 062.
4. Зберегти результат у `output-logs/5g_asterix.parquet`.
5. Обрати каталог з `.parquet`-датасетами.
6. Запустити навчання моделі у фоновому потоці.

Логи операцій відображаються у нижній частині вікна.

## 5G та ASTERIX через Bash

Для безперервного RF-демо:

```bash
./run_5g.sh
```

Or:

```bash
./asterion_5g.sh 5g
```

Демо працює до натискання `Ctrl+C`. Логи зберігаються у
`output-logs/rf-sensing.json`, графік сигналів — у `graphs/signals.png`.

Для кінцевої інтеграційної симуляції збереженням Parquet зручно використовувати
GUI. Міст програмно доступний через `integration.bridge.simulate_and_export`.

## Навчання моделі

Модель автоматично читає всі файли `training_*.parquet` із каталогу
`Datasets/` або з каталогу, переданого через `--data-dir`.

```bash
./run_model.sh train
```

The unified equivalent is:

```bash
./asterion_5g.sh train
```

З іншим каталогом:

```bash
./run_model.sh train --data-dir /path/to/parquet-data
```

Після навчання створюються:

- `model.keras` — TensorFlow-модель;
- `model_preprocessor.json` — стан кодувальника ознак.

Основна цільова колонка моделі — `TAXITIME_SEC_mvt`. Навчальні Parquet-файли
повинні містити цю колонку та часові/категоріальні поля, описані у
`Model/config_ml.py`.

## Прогнозування

```bash
./run_model.sh predict
```

Параметри за замовчуванням:

- ознаки: `Datasets/ranking.parquet`;
- шаблон результату: `Datasets/submitting.parquet`;
- модель: `model.keras`;
- препроцесор: `model_preprocessor.json`;
- результат: `submission.csv`.

Приклад із власними шляхами:

```bash
./run_model.sh predict \
  --model model.keras \
  --preprocessor model_preprocessor.json \
  --features Datasets/ranking.parquet \
  --template Datasets/submitting.parquet \
  --output submission.csv
```

## Оцінювання

```bash
./run_model.sh evaluate
```

За замовчуванням створюються метрики у stdout і графік
`feature_importance.png`.

## ASTERIX Parquet pipeline

Повний ASTERIX-пайплайн для Parquet запускається окремо:

```bash
cd yezhik-asterix
../.venv/bin/python run_pipeline.py \
  --input training_2025-01-01_2025-02-01.parquet \
  --threshold 1 \
  --interval 300 \
  --blind-spots-output blind_spots_for_simulation.csv \
  --simulation-output simulation_points.csv \
  --fused-output fused_dataset.parquet
```

Для ground tracks через OpenStreetMap додайте:

```bash
--ground-tracks --airport-icao EDDF \
--airport-place "Frankfurt Airport, Germany"
```

## Формати даних

Інтеграційний міст формує Parquet із полями:

- `icao24`, `timestamp`;
- `lat`, `lon`, `velocity`;
- `accuracy`, `range_m`, `azimuth_rad`, `elevation_rad`;
- `source`;
- `cat062_hex` — ASTERIX CAT 062 у hex-форматі.

Локальні координати симулятора конвертуються у приблизні WGS-84 координати
відносно початку `50.45, 30.52`. Це симуляційна прив'язка, а не геодезична
калібровка реального аеропорту.

## Перевірка

Перевірка Bash-скриптів:

```bash
bash -n run_app.sh run_5g.sh run_model.sh
```

Тести ASTERIX-пайплайна:

```bash
cd yezhik-asterix
../.venv/bin/pytest
```

## Примітки

- Кореневий `main.py` не використовується. Усі точки запуску — Bash-скрипти.
- Для запуску GUI потрібне встановлене системне tkinter-середовище.
- ASTERIX encoder використовує пакет `libasterix`/`ast-tool-py`.
- Навчання моделі та ASTERIX-обробка працюють із Parquet через `pandas` і
  `pyarrow`.
