import pandas as pd
import logging
import warnings
import os

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================
#  КОНФІГУРАЦІЯ — змінюйте лише цей блок
# ============================================================

FILE_PATH      = "КонтрагентАртикулКількість.csv"
ENCODING       = "utf-8-sig"
SEPARATOR      = ","
MIN_SUPPORT    = 0.002   # Зменшіть до 0.001, якщо результатів немає, частка товарів, що купуються спільно
MIN_CONFIDENCE = 0.2     # Якщо клієнт купив цільовий товар — яка ймовірність, що він купив і товар B
MIN_LIFT       = 2.0     # Тільки товари, що купуються разом частіше за випадковість
TARGET_PRODUCT = "DP0013"    # None = автоматично вибрати найпопулярніший

# ============================================================

COLUMN_SCHEMAS = {
    "schema_novyi4":     {"transaction": "Transaction_id", "product": "Product_Name"},
    "schema_kontragent": {"transaction": "user_id",        "product": "item_id"},
}

_SCHEMA_SIGNALS: dict[str, str] = {
    "transaction_id": "schema_novyi4",
    "product_name":   "schema_novyi4",
    "user_id":        "schema_kontragent",
    "item_id":        "schema_kontragent",
}


def detect_schema(columns: list[str]) -> dict | None:
    """Визначає схему стовпчиків за ключовими назвами."""
    cols_lower = {c.strip().lower() for c in columns}
    for signal, schema_key in _SCHEMA_SIGNALS.items():
        if signal in cols_lower:
            logging.info(f"Виявлено схему: {schema_key} (за колонкою '{signal}')")
            return COLUMN_SCHEMAS[schema_key]
    return None


def load_and_prep_data(file_path: str, encoding: str = "utf-8-sig", sep: str = ",") -> pd.DataFrame | None:
    """Завантажує CSV та повертає DataFrame з колонками Transaction_ID і Product_Name."""
    logging.info(f"Завантаження: {file_path}")

    if not os.path.exists(file_path):
        logging.error(f"Файл '{file_path}' не знайдено.")
        return None

    df = None
    for enc in (encoding, "cp1251"):
        try:
            df = pd.read_csv(file_path, encoding=enc, sep=sep, dtype=str)
            df.columns = df.columns.str.strip()
            logging.info(f"Прочитано (encoding={enc}). Колонки: {list(df.columns)}")
            break
        except UnicodeDecodeError:
            logging.warning(f"Кодування {enc!r} не підійшло, пробуємо наступне...")
        except Exception as e:
            logging.error(f"Помилка читання CSV: {e}")
            return None

    if df is None:
        logging.error("Не вдалося прочитати файл жодним кодуванням.")
        return None

    schema = detect_schema(list(df.columns))
    if schema is None:
        logging.error(
            f"Не вдалося розпізнати структуру. Знайдено: {list(df.columns)}\n"
            "Очікувались: 'Transaction_id'+'Product_Name' або 'user_id'+'item_id'"
        )
        return None

    tx_col, prod_col = schema["transaction"], schema["product"]
    missing = [c for c in (tx_col, prod_col) if c not in df.columns]
    if missing:
        logging.error(f"Відсутні обов'язкові колонки: {missing}")
        return None

    df = (
        df[[tx_col, prod_col]]
        .rename(columns={tx_col: "Transaction_ID", prod_col: "Product_Name"})
        .apply(lambda col: col.str.strip())
        .replace("", pd.NA)
        .dropna()
        .reset_index(drop=True)
    )

    logging.info(
        f"Підготовлено {len(df)} записів | "
        f"Клієнтів: {df['Transaction_ID'].nunique()} | "
        f"Товарів: {df['Product_Name'].nunique()}"
    )
    return df


def get_recommendations(
    df: pd.DataFrame,
    target_product: str,
    min_support: float = 0.005,
    min_confidence: float = 0.05,
    min_lift: float = 1.0,
) -> list[dict]:
    """
    Пряме обчислення метрик basket-аналізу для цільового товару.

    Замість повного перебору комбінацій (FP-Growth):
      1. Знаходимо транзакції з target_product.
      2. Рахуємо co-occurrence кожного іншого товару у цих транзакціях.
      3. Обчислюємо support / confidence / lift напряму через pandas.

    Результат математично ідентичний правилам вигляду {target} → {B},
    але без генерації всіх можливих itemsets.
    """
    n_total = df["Transaction_ID"].nunique()

    # --- Крок 1: підтримка кожного товару (для lift) ---
    item_support: pd.Series = (
        df.groupby("Product_Name")["Transaction_ID"]
        .nunique()
        .div(n_total)
    )

    # Фільтрація рідкісних товарів до будь-яких обчислень
    frequent_items = item_support[item_support >= min_support].index
    df_filtered = df[df["Product_Name"].isin(frequent_items)]
    logging.info(f"Товарів після фільтрації (support>={min_support}): {len(frequent_items)}")

    # --- Крок 2: транзакції з target_product ---
    if target_product not in frequent_items:
        logging.warning(f"'{target_product}' не проходить поріг min_support={min_support}.")
        return []

    tx_with_target: set = set(
        df_filtered.loc[df_filtered["Product_Name"] == target_product, "Transaction_ID"]
    )
    support_target = len(tx_with_target) / n_total
    logging.info(f"Транзакцій з '{target_product}': {len(tx_with_target)} (support={support_target:.4f})")

    # --- Крок 3: co-occurrence інших товарів у цих транзакціях ---
    co_df = df_filtered[
        df_filtered["Transaction_ID"].isin(tx_with_target) &
        (df_filtered["Product_Name"] != target_product)
    ]

    co_counts: pd.Series = co_df.groupby("Product_Name")["Transaction_ID"].nunique()

    # --- Крок 4: метрики ---
    metrics = pd.DataFrame({
        "support":    co_counts / n_total,
        "confidence": co_counts / len(tx_with_target),
        "lift":       (co_counts / len(tx_with_target)) / item_support[co_counts.index],
    })

    result = (
        metrics
        .query("confidence >= @min_confidence and lift >= @min_lift")
        .sort_values("lift", ascending=False)
        .round(4)
        .reset_index()
        .rename(columns={"Product_Name": "product"})
        .to_dict(orient="records")
    )

    logging.info(f"Знайдено рекомендацій: {len(result)}")
    return result


def print_results(target_product: str, recommendations: list[dict]) -> None:
    """Виводить результати у табличному форматі."""
    print("\n" + "=" * 62)
    print(f"  РЕКОМЕНДАЦІЇ ДЛЯ ТОВАРУ: {target_product}")
    print("=" * 62)

    if not recommendations:
        print("  Рекомендацій не знайдено.")
        print("  Спробуйте зменшити MIN_SUPPORT, MIN_CONFIDENCE або MIN_LIFT.")
    else:
        print(f"  {'Товар':<30} {'Підтримка':>10} {'Довіра':>10} {'Ліфт':>8}")
        print("  " + "-" * 62)
        for rec in recommendations:
            print(f"  {rec['product']:<30} {rec['support']:>10} {rec['confidence']:>10} {rec['lift']:>8}")

    print("=" * 62 + "\n")


# ============================================================
#  ТОЧКА ВХОДУ
# ============================================================
if __name__ == "__main__":
    data = load_and_prep_data(FILE_PATH, encoding=ENCODING, sep=SEPARATOR)

    if data is None:
        logging.error("Завантаження даних не вдалося. Перевірте конфігурацію.")
        exit(1)

    target = TARGET_PRODUCT or data["Product_Name"].value_counts().index[0]
    if TARGET_PRODUCT is None:
        logging.info(f"TARGET_PRODUCT не вказано — обрано найпопулярніший: '{target}'")

    recommendations = get_recommendations(data, target, MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT)
    print_results(target, recommendations)
