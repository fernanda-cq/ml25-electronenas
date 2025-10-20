import pandas as pd
from pathlib import Path
import os
import numpy as np

# Configuración de paths
CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE / "../../datasets/customer_purchases/"

def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    fullfilename = os.path.abspath(file)
    return pd.read_csv(fullfilename)

def extract_customer_number(customer_id):
    """Extrae solo el número del customer_id: CUST_0001 -> 1"""
    if isinstance(customer_id, str) and customer_id.startswith('CUST_'):
        # Remover 'CUST_' y convertir a int para quitar ceros a la izquierda
        number_str = customer_id.replace('CUST_', '')
        return str(int(number_str))  # Convertir a int y luego a string para quitar ceros
    return customer_id

def join_customer_ids():
    print("🔄 Uniendo customer_ids...")
    
    # Leer archivos
    original_train = read_csv("customer_purchases_train")
    processed_train = read_csv("customer_purchases_train_final_preprocessed")
    
    print(f"📊 Original train shape: {original_train.shape}")
    print(f"📊 Processed train shape: {processed_train.shape}")
    
    # Obtener los customer_ids únicos del original y extraer números
    original_customer_ids = original_train['customer_id'].unique()
    original_customer_numbers = [extract_customer_number(cid) for cid in original_customer_ids]
    
    print(f"📊 Total de clientes únicos en original: {len(original_customer_numbers)}")
    print(f"📊 Ejemplo de IDs convertidos: {original_customer_numbers[:5]}")
    
    # Para las primeras filas (que corresponden a los positivos), usar los números originales
    customer_numbers_joined = []
    
    # Asignar números originales a las primeras 7289 filas
    for i in range(len(original_train)):
        if i < len(original_customer_numbers):
            customer_numbers_joined.append(original_customer_numbers[i])
        else:
            # Si hay más filas que clientes únicos, repetir los números
            customer_numbers_joined.append(original_customer_numbers[i % len(original_customer_numbers)])
    
    # Para las filas adicionales (500 negativos), generar nuevos números empezando desde 501
    additional_rows = len(processed_train) - len(original_train)
    print(f"📊 Filas adicionales (negativos): {additional_rows}")
    
    # Generar nuevos números para los clientes negativos (501, 502, ...)
    start_new_id = 501
    new_customer_numbers = [str(i) for i in range(start_new_id, start_new_id + additional_rows)]
    
    # Combinar: números originales + nuevos números para negativos
    final_customer_numbers = customer_numbers_joined + new_customer_numbers
    
    print(f"📊 Total de customer_numbers generados: {len(final_customer_numbers)}")
    print(f"📊 Primeros 3 números: {final_customer_numbers[:3]}")
    print(f"📊 Últimos 3 números: {final_customer_numbers[-3:]}")
    
    # Agregar customer_id al processed y ponerlo al PRINCIPIO
    processed_with_ids = processed_train.copy()
    processed_with_ids['customer_id'] = final_customer_numbers
    
    # ✅ PONER customer_id AL PRINCIPIO de la lista de columnas
    columns = processed_with_ids.columns.tolist()
    columns.remove('customer_id')
    new_columns = ['customer_id'] + columns
    processed_with_ids = processed_with_ids[new_columns]
    
    # Verificar resultado
    print(f"✅ Processed con IDs shape: {processed_with_ids.shape}")
    print(f"✅ Customer_id es primera columna: {processed_with_ids.columns[0] == 'customer_id'}")
    print(f"✅ Primeras columnas: {processed_with_ids.columns.tolist()[:5]}")
    
    # Guardar resultado
    output_path = DATA_DIR / "customer_purchases_train_with_ids.csv"
    processed_with_ids.to_csv(output_path, index=False)
    print(f"💾 Archivo guardado en: {output_path}")
    
    return processed_with_ids

def join_test_customer_ids():
    print("\n🔄 Uniendo customer_ids para test...")
    
    # Leer archivos
    original_test = read_csv("customer_purchases_test")
    processed_test = read_csv("customer_purchases_test_final_preprocessed")
    
    print(f"📊 Original test shape: {original_test.shape}")
    print(f"📊 Processed test shape: {processed_test.shape}")
    
    # Extraer solo los números de los customer_ids del test (sin ceros)
    test_customer_numbers = [extract_customer_number(cid) for cid in original_test['customer_id']]
    
    # Para test, usar los números extraídos
    if len(original_test) == len(processed_test):
        processed_test_with_ids = processed_test.copy()
        processed_test_with_ids['customer_id'] = test_customer_numbers
        print("✅ Test: usando números extraídos directamente")
    else:
        print("⚠️  Test también tiene diferente número de filas")
        # Si son diferentes, usar la misma lógica que para train
        customer_numbers_to_join = []
        for i in range(len(processed_test)):
            if i < len(test_customer_numbers):
                customer_numbers_to_join.append(test_customer_numbers[i])
            else:
                # Generar nuevos números para test también si es necesario
                new_number = str(500 + i + 1)
                customer_numbers_to_join.append(new_number)
        
        processed_test_with_ids = processed_test.copy()
        processed_test_with_ids['customer_id'] = customer_numbers_to_join
    
    # ✅ PONER customer_id AL PRINCIPIO para test también
    columns_test = processed_test_with_ids.columns.tolist()
    columns_test.remove('customer_id')
    new_columns_test = ['customer_id'] + columns_test
    processed_test_with_ids = processed_test_with_ids[new_columns_test]
    
    print(f"✅ Test con IDs shape: {processed_test_with_ids.shape}")
    print(f"✅ Customer_id es primera columna test: {processed_test_with_ids.columns[0] == 'customer_id'}")
    print(f"✅ Primeros customer_numbers test: {processed_test_with_ids['customer_id'].head(3).tolist()}")
    
    # Guardar resultado
    output_path = DATA_DIR / "customer_purchases_test_with_ids.csv"
    processed_test_with_ids.to_csv(output_path, index=False)
    print(f"💾 Test con IDs guardado en: {output_path}")
    
    return processed_test_with_ids

if __name__ == "__main__":
    # Unir customer_ids para train
    train_with_ids = join_customer_ids()
    
    # Unir customer_ids para test
    test_with_ids = join_test_customer_ids()
    
    print("\n🎉 Proceso completado!")
    print(f"📁 Train con IDs: {len(train_with_ids)} filas, {train_with_ids['customer_id'].nunique()} clientes únicos")
    print(f"📁 Test con IDs: {len(test_with_ids)} filas, {test_with_ids['customer_id'].nunique()} clientes únicos")
    
    # Mostrar ejemplos del resultado
    print(f"\n📋 Ejemplo train (primeras 5 filas):")
    print(train_with_ids[['customer_id']].head(5).to_string(index=False))
    
    print(f"\n📋 Ejemplo test (primeras 5 filas):")
    print(test_with_ids[['customer_id']].head(5).to_string(index=False))