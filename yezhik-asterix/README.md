# ASTERIX Flight Data Pipeline

Пайплайн для аналізу даних польотів у Parquet, пошуку розривів у координатних треках і генерації синтетичних наземних маршрутів по taxiway-графу OpenStreetMap. Результати можна об'єднати та закодувати як ASTERIX CAT 062 edition 1.20.

## Можливості

- читає записи польотів із Parquet;
- нормалізує назви полів і підтримує типові поля `icao24`, `timestamp`, `lat`, `lon`;
- знаходить розриви між послідовними спостереженнями одного літака;
- лінійно інтерполює симуляційні точки між відомими координатами;
- опційно завантажує з OSM граф руліжних доріжок та маршрутизує від стоянки до смуги;
- обчислює середній час і віртуальну дистанцію для пар `STAND_mvt` / `RUNWAY_mvt`;
- об'єднує оригінальні та симуляційні записи;
- пакує координату WGS-84, час спостереження, трековий номер і вектор швидкості в CAT 062;
- має автоматичні тести для пошуку розривів, інтерполяції та об'єднання даних.

## Структура

```text
asterix_wrapper.py                  ASTERIX CAT 062 encoder
data_fusion.py                      Об'єднання оригінальних і симуляційних даних
find_blind_spots.py                 Пошук часових розривів у треках
generate_simulation_points.py       Генерація проміжних координат
taxiway_tracks.py                   OSM taxiway routing і синтетичні наземні треки
run_pipeline.py                     CLI для повного пайплайна
test_pipeline.py                    Автоматичні тести
test_taxiway_tracks.py              Тести OSM-маршрутів і CAT 062
requirements.txt                    Python-залежності
training_2025-01-01_2025-02-01.parquet  Вхідний тренувальний набір
```

Файли `blind_spots_for_simulation.csv`, `simulation_points.csv` і `fused_dataset.parquet` є результатами роботи пайплайна та можуть бути перегенеровані.

## Вимоги

- Windows PowerShell;
- Python 3.10 або новіший;
- пакети з `requirements.txt`;
- доступ до OpenStreetMap Overpass API для першого завантаження карти в режимі ground tracks.

## Встановлення

Створіть віртуальне середовище та активуйте його:

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Встановіть основні залежності:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Запуск

Запуск стандартної конфігурації:

```bash
python run_pipeline.py \
  --input training_2025-01-01_2025-02-01.parquet `
  --threshold 1 `
  --interval 300 `
  --blind-spots-output blind_spots_for_simulation.csv `
  --simulation-output simulation_points.csv `
  --fused-output fused_dataset.parquet
```

Параметри:

| Параметр | Значення за замовчуванням | Призначення |
|---|---:|---|
| `--input` | `training_2025-01-01_2025-02-01.parquet` | Вхідний Parquet-файл |
| `--threshold` | `1` | Мінімальний розрив у секундах |
| `--interval` | `300` | Інтервал між симуляційними точками |
| `--blind-spots-output` | `blind_spots_for_simulation.csv` | CSV знайдених розривів |
| `--simulation-output` | `simulation_points.csv` | CSV симуляційних точок |
| `--fused-output` | `fused_dataset.parquet` | Підсумковий Parquet-файл |
| `--ground-tracks` | вимкнено | Увімкнути генерацію маршрутів по OSM taxiway-графу |
| `--airport-icao` | `EDDF` | ICAO аеропорту для відбору локальних вильотів/прибуттів |
| `--airport-place` | `Frankfurt Airport, Germany` | Назва місця для OSMnx-запиту |
| `--pair-profiles-output` | `taxi_pair_profiles.csv` | Середні taxi-time та віртуальні дистанції пар |

OSM-режим для Frankfurt Airport:

```bash
python run_pipeline.py \
  --input training_2025-01-01_2025-02-01.parquet `
  --ground-tracks `
  --airport-icao EDDF `
  --airport-place "Frankfurt Airport, Germany" `
  --interval 30 `
  --simulation-output ground_tracks.csv `
  --pair-profiles-output taxi_pair_profiles.csv `
  --fused-output fused_dataset.parquet
```

Для іншого аеропорту потрібно одночасно змінити `--airport-icao` та `--airport-place`. Один запуск обробляє один аеропорт.

## Алгоритм

Звичайний режим шукає часові розриви та інтерполює між наявними координатами. OSM-режим є окремим:

1. Дані відбираються за ICAO локального аеропорту (`ADEP_mvt` для DEP, `ADES_mvt` для ARR).
2. Для кожної пари стоянка-смуга рахується середній додатний `TAXITIME_SEC_mvt`. Довідкова віртуальна дистанція дорівнює `mean_taxi_seconds * 4 м/с`; це масштаб для порівняння пар, не виміряна довжина руліжної доріжки.
3. OSMnx завантажує `aeroway=taxiway`, `aeroway=parking_position` і `aeroway=runway`. Між найближчими вузлами будується найкоротший маршрут за довжиною, використовуючи геометрії ребер графа.
4. Для окремого руху використовується його валідний `TAXITIME_SEC_mvt`; якщо він некоректний, застосовується середній час пари з поправкою на категорію літака.
5. Час розподіляється вздовж маршруту з легким сповільненням біля кінців; тип літака змінює цей профіль. Загальна тривалість валідного індивідуального `TAXITIME_SEC_mvt` при цьому лишається сталою. Для DEP маршрут іде stand→runway перед `MVT_TIME_UTC_mvt`; для ARR runway→stand після цієї часової мітки.
6. Координати, час, трековий номер і `vx`/`vy` пакуються в CAT 062 edition 1.20.

## Важливе обмеження вхідних даних

Файл `training_2025-01-01_2025-02-01.parquet` не містить фактичних `lat` і `lon`. OSM-режим синтезує правдоподібний трек за топологією карти й часовими полями, але це не виміряний трек літака.

Маршрут генерується тільки вздовж taxiway-ребер після прив'язки stand/runway до найближчого вузла. Запис пропускається, якщо OSM не має відповідного `ref`/`name`, вузол надто далеко або граф не має маршруту. Назви стоянок і смуг у OSM можуть відрізнятися від датасету; відповідність перевіряється за тегами `ref`, `name`, `local_ref` тощо.

Для виміряного, а не синтетичного треку потрібен вхідний набір із позиційними полями:

```text
icao24, timestamp, lat, lon
```

У режимі OSM якість залежить від покриття та точності OpenStreetMap, зіставлення ідентифікаторів, похибки прив'язки та евристики часової моделі. Вектор швидкості є похідним від синтетичних точок. Вихід не слід трактувати як точні дані surveillance/MLAT або як operational navigation data.

Дані карти © OpenStreetMap contributors, ліцензія Open Database License (ODbL). Під час поширення похідних даних дотримуйтеся вимог атрибуції та ODbL.

## Тести

```powershell
python -m pytest -q
```

Тести використовують тимчасові Parquet/CSV-файли та не змінюють тренувальний набір.

## Формат результатів

`blind_spots_for_simulation.csv` містить попередню та поточну часові мітки, координати і тривалість розриву.

`simulation_points.csv` містить згенеровані поля `lat`, `lon`, `timestamp`, `velocity` і `source=simulated`.

`fused_dataset.parquet` містить оригінальні та симуляційні записи, а також `cat062_hex`.

## Ліцензія

Ліцензію проєкту ще не визначено.