import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from pandas.tseries.offsets import MonthEnd
from dateutil.relativedelta import relativedelta
import openai
import boto3
from io import BytesIO


conn_str = st.secrets["MYSQL_CONNECTION_STRING"]
openai.api_key = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=openai.api_key)

def calculate_entropy(values):
    total = sum(values)
    probabilities = [v / total for v in values]
    return -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)


def resumen_flujo_neto_y_ventas(rfc_key):
    engine = create_engine(conn_str)

    hoy = datetime.now()
    if hoy.day < 28:
        mes_analisis = hoy.replace(day=1) - timedelta(days=1)
    else:
        mes_analisis = hoy

    nombre_mes = mes_analisis.strftime('%B %Y')
    fecha_fin = datetime(mes_analisis.year, mes_analisis.month, 1) + pd.offsets.MonthEnd(0)
    fecha_inicio = fecha_fin - pd.DateOffset(years=1)

    q_cli = f"""
    SELECT issuedAt, totalMxn 
    FROM invoices 
    WHERE isIssuer = 1 AND status = 'VIGENTE' AND issuerRfc = '{rfc_key}'
    """
    df_cli = pd.read_sql(q_cli, engine)
    df_cli['issuedAt'] = pd.to_datetime(df_cli['issuedAt'], errors='coerce')
    df_cli = df_cli[(df_cli['issuedAt'] >= fecha_inicio) & (df_cli['issuedAt'] <= fecha_fin)]
    df_cli['month_year'] = df_cli['issuedAt'].dt.to_period('M')
    df_cli = df_cli.groupby('month_year')['totalMxn'].sum().reset_index()
    df_cli['month_year'] = df_cli['month_year'].astype(str)

    q_prov = f"""
    SELECT issuedAt, total 
    FROM invoices 
    WHERE isReceiver = 1 AND status = 'VIGENTE' AND receiverRfc = '{rfc_key}'
    """
    df_prov = pd.read_sql(q_prov, engine)
    df_prov['issuedAt'] = pd.to_datetime(df_prov['issuedAt'], errors='coerce')
    df_prov = df_prov[(df_prov['issuedAt'] >= fecha_inicio) & (df_prov['issuedAt'] <= fecha_fin)]
    df_prov['month_year'] = df_prov['issuedAt'].dt.to_period('M')
    df_prov = df_prov.groupby('month_year')['total'].sum().reset_index()
    df_prov['month_year'] = df_prov['month_year'].astype(str)

    df = pd.merge(df_cli, df_prov, on='month_year', how='outer').fillna(0)
    df.columns = ['month_year', 'ingresos', 'egresos']
    df['flujo_neto'] = df['ingresos'] - df['egresos']
    df['month_year'] = pd.to_datetime(df['month_year'], format='%Y-%m')
    df = df.sort_values('month_year').reset_index(drop=True)

    if len(df) < 2:
        print("No hay suficientes datos para generar el resumen.")
        return

    row_actual = df.iloc[-1]
    row_anterior = df.iloc[-2]

    ingresos = row_actual['ingresos']
    egresos = row_actual['egresos']
    flujo_neto = row_actual['flujo_neto']
    flujo_neto_anterior = row_anterior['flujo_neto']
    diferencia_absoluta = flujo_neto - flujo_neto_anterior
    cambio_pct = ((diferencia_absoluta) / flujo_neto_anterior * 100) if flujo_neto_anterior != 0 else float('inf')

    resumen_flujo = f"""
📊 En el mes de {nombre_mes}, la empresa con RFC {rfc_key} recibió ingresos por ${ingresos:,.2f} y realizó pagos por ${egresos:,.2f}, 
generando un flujo neto de ${flujo_neto:,.2f}. El flujo neto del mes anterior fue de ${flujo_neto_anterior:,.2f}, lo que representa una 
diferencia de ${diferencia_absoluta:,.2f} y un cambio del {cambio_pct:.2f}% en comparación con el mes anterior.
"""
    engine.dispose()
    return resumen_flujo


def resumen_proveedores(rfc_key):
    engine = create_engine(conn_str)

    hoy = datetime.now()
    if hoy.day < 28:
        mes_analisis = hoy.replace(day=1) - timedelta(days=1)
    else:
        mes_analisis = hoy

    mes = mes_analisis.month
    año = mes_analisis.year
    nombre_mes = mes_analisis.strftime('%B %Y')

    start_date = datetime(año, mes, 1)
    if mes == 12:
        end_date = datetime(año + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(año, mes + 1, 1) - timedelta(days=1)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    pagos_query = f"""
        SELECT issuedAt, total
        FROM invoices 
        WHERE isReceiver = 1 AND status = 'VIGENTE' AND receiverRfc = '{rfc_key}'
    """
    df_pagos = pd.read_sql(pagos_query, engine)
    df_pagos['issuedAt'] = pd.to_datetime(df_pagos['issuedAt'], errors='coerce')
    df_pagos = df_pagos[df_pagos['issuedAt'].notna()]
    df_pagos['mes'] = df_pagos['issuedAt'].dt.month
    df_pagos['anio'] = df_pagos['issuedAt'].dt.year

    if mes == 1:
        mes_anterior = 12
        año_anterior = año - 1
    else:
        mes_anterior = mes - 1
        año_anterior = año

    df_actual = df_pagos[(df_pagos['mes'] == mes) & (df_pagos['anio'] == año)]
    df_anterior = df_pagos[(df_pagos['mes'] == mes_anterior) & (df_pagos['anio'] == año_anterior)]

    total_actual = df_actual['total'].sum()
    total_anterior = df_anterior['total'].sum()
    cambio_pct = ((total_actual - total_anterior) / total_anterior * 100) if total_anterior != 0 else float('inf')

    proveedores_query = f"""
        SELECT issuedAt, issuerName, issuerRfc, total
        FROM invoices 
        WHERE isReceiver = 1 AND status = 'VIGENTE' 
        AND receiverRfc = '{rfc_key}' 
        AND issuedAt BETWEEN '{start_str}' AND '{end_str}'
    """
    df_prov = pd.read_sql(proveedores_query, engine)
    df_prov['total'] = df_prov['total'].astype(float)

    total_por_prov = df_prov.groupby('issuerRfc')['total'].sum().reset_index()
    total_por_prov = pd.merge(total_por_prov, df_prov[['issuerRfc', 'issuerName']].drop_duplicates(), on='issuerRfc')

    top_5 = total_por_prov.nlargest(5, 'total')
    entropy = calculate_entropy(total_por_prov['total'])
    max_entropy = np.log2(len(total_por_prov)) if len(total_por_prov) > 1 else 1
    kpr = (entropy / max_entropy) * 100

    top_provider_names = ', '.join(top_5['issuerName'].tolist())

    rfcs_12 = df_prov[df_prov['issuerRfc'].str.len() == 12]['issuerRfc'].nunique()
    rfcs_13 = df_prov[df_prov['issuerRfc'].str.len() == 13]['issuerRfc'].nunique()


    resumen_proveedores = f"""
📦 En el mes de {nombre_mes}, la empresa con RFC {rfc_key} realizó pagos a proveedores por un total de ${total_actual:,.2f}, 
lo que representa un cambio del {cambio_pct:.2f}% respecto al mes anterior. 
Las compras provinieron principalmente de: {top_provider_names}, con una distribución de pagos del {kpr:.2f}% según la entropía.
Se identificaron {rfcs_12} empresas (RFC 12) y {rfcs_13} personas físicas (RFC 13) como emisores de las facturas.
"""
    engine.dispose()
    
    return resumen_proveedores




def reemplazar_publico_general(df):
    df.loc[df['receiverRfc'] == 'XAXX010101000', 'receiverName'] = 'PUBLICO GENERAL'
    return df

def reemplazar_extranjero(df):
    df.loc[df['receiverRfc'] == 'XEXX010101000', 'receiverName'] = 'EXTRANJEROS'
    return df

def calculate_entropy(values):
    total = sum(values)
    probabilities = [v / total for v in values]
    entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)
    return entropy

def resumen_empresa(rfc_key):
    engine = create_engine(conn_str)

    hoy = pd.Timestamp.now()
    ultimo_mes_completo = hoy - MonthEnd(1)
    mes_final = ultimo_mes_completo.month
    anio_final = ultimo_mes_completo.year

    mes_anterior = mes_final - 1
    anio_anterior = anio_final
    if mes_anterior == 0:
        mes_anterior = 12
        anio_anterior -= 1

    nombre_mes = datetime(anio_final, mes_final, 1).strftime('%B %Y')

    three_months_ago = hoy - timedelta(days=90)
    query_status = f"""
    SELECT issuedAt, status
    FROM invoices 
    WHERE isIssuer = 1 
    AND issuerRfc = '{rfc_key}' 
    AND issuedAt BETWEEN '{three_months_ago.strftime('%Y-%m-%d')}' AND '{hoy.strftime('%Y-%m-%d')}'
    """
    df_cancel = pd.read_sql(query_status, engine)
    df_cancel['issuedAt'] = pd.to_datetime(df_cancel['issuedAt'])
    total_3m = len(df_cancel)
    canceladas_3m = len(df_cancel[df_cancel['status'] == "CANCELADO"])
    pct_cancel_3m = (canceladas_3m / total_3m * 100) if total_3m > 0 else 0

    df_mes = df_cancel[(df_cancel['issuedAt'].dt.month == mes_final) & (df_cancel['issuedAt'].dt.year == anio_final)]
    total_mes = len(df_mes)
    canceladas_mes = len(df_mes[df_mes['status'] == "CANCELADO"])
    pct_cancel_mes = (canceladas_mes / total_mes * 100) if total_mes > 0 else 0

    query_ventas = f'''
    SELECT issuedAt, totalMxn, creditedAmount
    FROM invoices 
    WHERE isIssuer = 1 AND status = "VIGENTE" AND issuerRfc = "{rfc_key}" AND issuedAt >= '2023-01-01'
    '''
    df_ventas = pd.read_sql(query_ventas, engine)
    df_ventas['issuedAt'] = pd.to_datetime(df_ventas['issuedAt'], errors='coerce')
    df_ventas = df_ventas[df_ventas['issuedAt'].notna()]
    df_ventas['mes'] = df_ventas['issuedAt'].dt.month
    df_ventas['anio'] = df_ventas['issuedAt'].dt.year

    total_por_mes = df_ventas.groupby(['anio', 'mes'])['totalMxn'].sum()
    credito_por_mes = df_ventas.groupby(['anio', 'mes'])['creditedAmount'].sum()
    ventas_netas = total_por_mes - credito_por_mes

    total_act = ventas_netas.get((anio_final, mes_final), 0)
    total_ant = ventas_netas.get((anio_anterior, mes_anterior), 0)
    cambio_pct = ((total_act - total_ant) / total_ant * 100) if total_ant else float('nan')

    # --- CLIENTES POR MES ACTUAL ---
    start_date = datetime(anio_final, mes_final, 1)
    end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    query_clientes = f'''
    SELECT issuedAt, receiverName, receiverRfc, totalMxn 
    FROM invoices 
    WHERE isIssuer = 1 
    AND status = "VIGENTE" 
    AND issuerRfc = "{rfc_key}" 
    AND issuedAt BETWEEN "{start_str}" AND "{end_str}"
    '''
    df_clientes = pd.read_sql(query_clientes, engine)
    engine.dispose()

    df_clientes = reemplazar_publico_general(df_clientes)
    df_clientes = reemplazar_extranjero(df_clientes)
    df_clientes['totalMxn'] = df_clientes['totalMxn'].astype(float)

    total_por_cliente = df_clientes.groupby('receiverRfc')['totalMxn'].sum().reset_index()
    total_por_cliente = pd.merge(total_por_cliente, df_clientes[['receiverRfc', 'receiverName']].drop_duplicates(), on='receiverRfc')
    
    total_mes_actual = total_por_cliente['totalMxn'].sum()
    total_por_cliente['porcentaje'] = (total_por_cliente['totalMxn'] / total_mes_actual) * 100

    top_5 = total_por_cliente.nlargest(5, 'totalMxn')

    entropy = calculate_entropy(total_por_cliente['totalMxn'])
    max_entropy = np.log2(len(total_por_cliente)) if len(total_por_cliente) > 1 else 1
    kpr = (entropy / max_entropy) * 100

    top_clientes_format = ', '.join([f"{row['receiverName']}: {row['porcentaje']:.1f}%" for _, row in top_5.iterrows()])

    resumen_clientes = f"""
📊 En el mes de {nombre_mes}, la empresa con RFC {rfc_key} tuvo ventas por ${total_act:,.2f}, 
lo que representa un cambio del {cambio_pct:.2f}% respecto al mes anterior. 
En cuanto a cancelaciones, el {pct_cancel_3m:.2f}% de las facturas fueron canceladas en los últimos tres meses 
y el {pct_cancel_mes:.2f}% en el mes de análisis. Los ingresos provinieron principalmente de: 
{top_clientes_format}, con una distribución de ventas diversificada en un {kpr:.2f}% según la métrica de entropía.
"""

    return resumen_clientes

def return_empleados(rfc_key):
    engine = create_engine(conn_str)
    query_1 = "SELECT id, total, date FROM employees WHERE id LIKE %s"
    df = pd.read_sql(query_1, engine, params=(f'{rfc_key}-%',))
    engine.dispose()

    df['date_from_id'] = df['id'].str.extract(r'-(\d{4}-\d{2})$')[0]
    df['id'] = df['id'].str.extract(r'(^.+)-\d{4}-\d{2}')[0]
    df['date'] = pd.to_datetime(df['date'])

    current_date = datetime.now()
    one_year_ago = current_date - timedelta(days=365)

    df_filtered = df.loc[(df['date'] >= one_year_ago) & (df['date'].dt.month != current_date.month)].copy()
    df_filtered['month'] = df_filtered['date'].dt.strftime('%Y-%m')

    df_monthly = df_filtered.groupby('month')['total'].sum().reset_index()
    df_monthly['change'] = df_monthly['total'].pct_change() * 100  

    std_total = df_monthly['total'].std()
    mean_total = df_monthly['total'].mean()

    # Evaluaciones
    cambio_empleados = df_monthly['change'].iloc[-1]
    cambio_texto = ""
    if cambio_empleados > 10:
        cambio_texto = f"⚠️ El número de empleados aumentó un {cambio_empleados:.2f}% en el último mes."
    elif cambio_empleados < -10:
        cambio_texto = f"⚠️ El número de empleados disminuyó un {abs(cambio_empleados):.2f}% en el último mes."
    else:
        cambio_texto = "El número de empleados se ha mantenido estable en el último mes."

    variacion_texto = ""
    if std_total > mean_total * 0.25:
        variacion_texto = f"⚠️ Además, se detecta una variación significativa en el número de empleados (desviación estándar: {std_total:.2f})."

    resumen = f"""
👥 Análisis de Empleados para el RFC {rfc_key}:
{cambio_texto} {variacion_texto}
"""
    return resumen

def resumen_financieras(rfc_key):
    aws_access_key = st.secrets["aws"]["aws_access_key"]
    aws_secret_key = st.secrets["aws"]["aws_secret_key"]
    region_name = st.secrets["aws"]["region_name"]
    bucket_name = st.secrets["aws"]["bucket_name"]
    file_key = st.secrets["aws"]["file_key"]

    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region_name
    )
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    file_stream = BytesIO(response['Body'].read())
    excel_data = pd.ExcelFile(file_stream, engine='openpyxl')

    today = datetime.now()
    end_date = today.replace(day=1) - pd.DateOffset(days=1)
    start_date = end_date.replace(day=1)
    periodo_texto = end_date.strftime('%B %Y')

    engine = create_engine(conn_str)

    def procesar_institucion(sheet_name, tipo_institucion):
        sheet_data = excel_data.parse(sheet_name)
        rfc_list = sheet_data['RFC'].tolist()

        query = f"""
        SELECT issuerRfc, issuerName, total, issuedAt
        FROM invoices
        WHERE isReceiver = 1
        AND receiverRfc = '{rfc_key}'
        AND status = 'VIGENTE'
        """
        invoices_data = pd.read_sql(query, engine)
        invoices_data['issuedAt'] = pd.to_datetime(invoices_data['issuedAt'], errors='coerce')

        invoices_data = invoices_data[invoices_data['issuerRfc'].isin(rfc_list)]
        filtered_data = invoices_data[
            (invoices_data['issuedAt'] >= start_date) &
            (invoices_data['issuedAt'] <= end_date)
        ]

        resumen_df = filtered_data.groupby('issuerName')['total'].sum().reset_index()
        resumen_df = resumen_df.sort_values(by='total', ascending=False)

        if resumen_df.empty:
            return f"En {periodo_texto}, no se registraron transacciones con {tipo_institucion}s."

        lista_texto = [f"{row['issuerName']}: ${row['total']:,.2f}" for _, row in resumen_df.iterrows()]
        lista_resumida = ', '.join(lista_texto)

        return f"🏦 Transacciones con {tipo_institucion}s durante {periodo_texto}: {lista_resumida}."

    bancos = procesar_institucion("Bancos", "banco")
    sofipos = procesar_institucion("SOFIPO", "SOFIPO")
    sofomes = procesar_institucion("SOFOM", "SOFOM")

    engine.dispose()

    return f"""
— Transacciones Financieras —
{bancos}

{sofipos}

{sofomes}
"""
    
def generar_analisis_gpt(rfc_key: str) -> str:
    flujo = resumen_flujo_neto_y_ventas(rfc_key)
    prov  = resumen_proveedores(rfc_key)
    emp   = resumen_empresa(rfc_key)
    emple = empleados(rfc_key)  # <- Ahora se incluye aquí

    prompt = f"""
Empresa (RFC: {rfc_key}):

— Flujo Neto y Ventas —
{flujo}

— Proveedores —
{prov}

— Clientes y Cancelaciones —
{emp}

— Empleados —
{emple}

{fin}

Como analista financiero experto, haz un análisis unificado:
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Eres un analista financiero experto."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1024
    )

    return response.choices[0].message.content

    
engine = create_engine(conn_str)

@st.cache_data(show_spinner=False)
def get_clients():
    df = pd.read_sql("SELECT rfc, name FROM clients", engine)
    return df
    
st.title("Análisis Financiero por Cliente")
dclients = get_clients()
sel = st.selectbox("Cliente:", dclients["name"])
rfc = dclients.loc[dclients.name==sel, "rfc"].iloc[0]

for label, fn in [
    ("Flujo Neto y Ventas", resumen_flujo_neto_y_ventas),
    ("Proveedores", resumen_proveedores),
    ("Clientes y Cancelaciones", resumen_empresa),
    ("Empleados", empleados), 
    ("Transacciones Financieras", resumen_financieras), 
]:
    with st.expander(f"🔹 {label}"):
        st.markdown(fn(rfc))

    
if st.button("🧠 Generar Análisis Chat"):
    with st.spinner("Pensando…"):
        st.markdown("### Análisis Unificado por GPT")
        st.write(generar_analisis_gpt(rfc))
