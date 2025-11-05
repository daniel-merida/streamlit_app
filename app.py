import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Análise Exploratória - Streamlit", layout="wide")

st.title("Análise Exploratória e Pré-processamento")
st.write("Faça upload de um CSV de vendas; o app executa as funções do seu notebook e gera um Excel com várias abas.")

# -----------------------------
# Copia das funções principais (adaptadas do notebook)
# -----------------------------

pd.set_option('display.float_format', '{:,.2f}'.format)

# FUNÇÕES: venda_zero_detalhe_pdv / resumo_canal / segmentacao_pareto_df / detectar_outliers_multidelta / venda_zero

def venda_zero_detalhe_pdv(df, limites=[1, 2, 3],
                            col_client='Client_ID', col_mes='Mes', col_valor='Valor',
                            col_channel='Channel', col_subchannel='Sub_Channel'):
    sales_df = df.copy()
    sales_df[col_mes] = sales_df[col_mes].astype(str).str.zfill(6)
    sales_df['ano'] = pd.to_datetime(sales_df[col_mes], format='%Y%m', errors='coerce').dt.year
    sales_df['mes_num'] = pd.to_datetime(sales_df[col_mes], format='%Y%m', errors='coerce').dt.month
    sales_df['data'] = pd.to_datetime(dict(year=sales_df['ano'], month=sales_df['mes_num'], day=1))
    data_max = sales_df['data'].max()

    registros = []
    for limite in limites:
        months_to_check = [data_max - pd.DateOffset(months=i) for i in range(0, limite)]
        df_filtro = sales_df[sales_df['data'].isin(months_to_check) & (sales_df[col_valor] <= 0)].copy()
        cont_por_cliente = (
            df_filtro
            .groupby(col_client)['data']
            .nunique()
            .reset_index(name='count_months')
        )
        clientes_selecionados = cont_por_cliente.loc[
            cont_por_cliente['count_months'] == limite, col_client
        ].unique()
        df_clientes = sales_df[sales_df[col_client].isin(clientes_selecionados)][
            [col_client, col_channel, col_subchannel]
        ].drop_duplicates()
        for _, row in df_clientes.iterrows():
            registros.append((f"{int(limite)*30}", row[col_client], row[col_channel], row[col_subchannel]))

    df_resultado = pd.DataFrame(registros, columns=['Venda_Zero', 'Client_ID', 'Channel', 'Sub_Channel'])
    if df_resultado.empty:
        df_canal_vendazero = pd.DataFrame(columns=['Channel','Sub_Channel','VZ_30','VZ_60','VZ_90'])
    else:
        df_canal_vendazero = (
            df_resultado.groupby(['Channel', 'Sub_Channel', 'Venda_Zero'])['Client_ID']
            .nunique()
            .reset_index()
            .pivot(index=['Channel', 'Sub_Channel'], columns='Venda_Zero', values='Client_ID')
            .fillna(0)
            .astype(int)
            .reset_index()
        )
        df_canal_vendazero.columns.name = None
        df_canal_vendazero = df_canal_vendazero.rename(columns={
            '30': 'VZ_30', '60': 'VZ_60', '90': 'VZ_90'
        })

    return df_canal_vendazero, df_resultado


def resumo_canal(df, incluir_venda_zero=True):
    df_cliente = df.groupby(['Client_ID', 'Channel', 'Sub_Channel'])['Valor'].sum().reset_index()
    df_grouped = df_cliente.groupby(['Channel', 'Sub_Channel']).agg(
        Qtd_PDV=('Client_ID', 'nunique'),
        Volume_Total=('Valor', 'sum'),
        Volume_Avg_PDV=('Valor', 'mean'),
        Volume_Std_PDV=('Valor', 'std')
    ).reset_index()

    if incluir_venda_zero:
        df_vz, df_vz_resultado = venda_zero_detalhe_pdv(df)
        df_grouped = df_grouped.merge(df_vz, on=['Channel', 'Sub_Channel'], how='left')
        for col in ['VZ_30', 'VZ_60', 'VZ_90']:
            if col not in df_grouped.columns:
                df_grouped[col] = 0
        df_grouped[['VZ_30', 'VZ_60', 'VZ_90']] = df_grouped[['VZ_30', 'VZ_60', 'VZ_90']].fillna(0).astype(int)

    return df_grouped


def segmentacao_pareto_df(sales_df, col_client='Client_ID', col_valor='Valor', col_channel='Channel'):
    sales_df = sales_df.copy()
    # tenta garantir coluna Mes string AAAAMM
    sales_df['Mes'] = sales_df['Mes'].astype(str).str.zfill(6)
    sales_df['ano'] = pd.to_datetime(sales_df['Mes'], format='%Y%m', errors='coerce').dt.year
    sales_df['mes'] = pd.to_datetime(sales_df['Mes'], format='%Y%m', errors='coerce').dt.month

    sales_seg_by_month_df = sales_df.groupby([col_client, 'ano', 'mes'])[col_valor].sum().reset_index()
    sales_seg_by_client_df = sales_seg_by_month_df.groupby(col_client)[col_valor].sum().reset_index()

    # months active
    sales_seg_by_client_df['months'] = sales_seg_by_month_df.groupby(col_client).size().reset_index(drop=True)
    sales_seg_by_client_df['unit_cases'] = (sales_seg_by_client_df[col_valor] / sales_seg_by_client_df['months']) * 12

    total_unit_cases = sales_seg_by_client_df['unit_cases'].sum()
    if total_unit_cases == 0:
        sales_seg_by_client_df['sales_percentage'] = 0
    else:
        sales_seg_by_client_df['sales_percentage'] = sales_seg_by_client_df['unit_cases'] / total_unit_cases

    sales_seg_by_client_df = sales_seg_by_client_df.sort_values('sales_percentage', ascending=False).reset_index(drop=True)
    sales_seg_by_client_df['acc_percentages'] = sales_seg_by_client_df['sales_percentage'].cumsum()

    for p, seg in zip([0.2*i for i in range(5)], ['A','B','C','D','E']):
        sales_seg_by_client_df.loc[sales_seg_by_client_df['acc_percentages'] > p, 'New_Segment'] = seg

    sales_seg_by_client_df = sales_seg_by_client_df.merge(
        sales_df[[col_client, col_channel]].drop_duplicates(),
        how='left',
        on=col_client
    )

    resumo = (
        sales_seg_by_client_df
        .groupby(['New_Segment','Channel'])
        .agg(
            Qtd_PDV=('Client_ID','nunique'),
            Volume_Total=(col_valor,'sum'),
            Volume_Avg_PDV=(col_valor,'mean')
        )
        .reset_index()
        .sort_values(['Channel','New_Segment'])
    )

    resumo['Mix_PDV'] = resumo['Qtd_PDV'] / resumo.groupby('Channel')['Qtd_PDV'].transform('sum') * 100
    resumo['Mix_Volume'] = resumo['Volume_Total'] / resumo.groupby('Channel')['Volume_Total'].transform('sum') * 100
    resumo['Mix_PDV'] = resumo['Mix_PDV'].round(2)
    resumo['Mix_Volume'] = resumo['Mix_Volume'].round(2)

    return resumo


def detectar_outliers_multidelta(
    df,
    deltas=[1.5, 2, 2.5, 3],
    col_client='Client_ID',
    col_valor='Valor',
    col_channel='Channel',
    col_segment='New_Segment',
    col_mes='Mes',
    col_ko_class='KO_Class',
    col_sub_channel='Sub_Channel',
    order_general=None,
    order_detalhe=None
):
    sales_df = df.copy()
    mes_series = sales_df[col_mes].astype(str).str.zfill(6)
    global_data_max = pd.to_datetime(mes_series, format='%Y%m', errors='coerce').max()
    sales_df['Mes'] = mes_series
    sales_df['data'] = pd.to_datetime(sales_df['Mes'], format='%Y%m', errors='coerce')
    sales_df['ano'] = sales_df['data'].dt.year
    sales_df['mes'] = sales_df['data'].dt.month

    sales_seg_by_month_df = sales_df.groupby([col_client, 'ano', 'mes'])[col_valor].sum().reset_index()
    sales_seg_by_client_df = sales_seg_by_month_df.groupby(col_client)[col_valor].sum().reset_index()
    sales_seg_by_client_df['months'] = sales_seg_by_month_df.groupby(col_client).size().reset_index(drop=True)
    sales_seg_by_client_df['unit_cases'] = (sales_seg_by_client_df[col_valor] / sales_seg_by_client_df['months']) * 12

    total_unit_cases = sales_seg_by_client_df['unit_cases'].sum()
    if total_unit_cases == 0:
        sales_seg_by_client_df['sales_percentage'] = 0
    else:
        sales_seg_by_client_df['sales_percentage'] = sales_seg_by_client_df['unit_cases'] / total_unit_cases

    sales_seg_by_client_df = sales_seg_by_client_df.sort_values('sales_percentage', ascending=False).reset_index(drop=True)
    sales_seg_by_client_df['acc_percentages'] = sales_seg_by_client_df['sales_percentage'].cumsum()
    for p, seg in zip([0.2*i for i in range(5)], ['A', 'B', 'C', 'D', 'E']):
        sales_seg_by_client_df.loc[sales_seg_by_client_df['acc_percentages'] > p, col_segment] = seg

    merge_cols = [col_client, col_channel]
    if col_ko_class in sales_df.columns:
        merge_cols.append(col_ko_class)
    if col_sub_channel in sales_df.columns:
        merge_cols.append(col_sub_channel)

    sales_seg_by_client_df = sales_seg_by_client_df.merge(
        sales_df[merge_cols].drop_duplicates(),
        how='left', on=col_client
    )

    df_agg = sales_seg_by_client_df.groupby([col_client, col_channel, col_segment])[col_valor].sum().reset_index()

    detalhe_group_cols = [col_client, col_channel, col_segment]
    if col_ko_class in sales_seg_by_client_df.columns:
        detalhe_group_cols.append(col_ko_class)
    if col_sub_channel in sales_seg_by_client_df.columns:
        detalhe_group_cols.append(col_sub_channel)

    df_agg_detalhe = sales_seg_by_client_df.groupby(detalhe_group_cols)[col_valor].sum().reset_index()

    metrics = df_agg.groupby([col_channel, col_segment])[col_valor].agg(
        Volume_Total='sum',
        Volume_Avg_PDV='mean',
        Volume_Std_PDV='std'
    ).reset_index()

    df_agg = df_agg.merge(metrics, on=[col_channel, col_segment], how='left')
    df_agg_detalhe = df_agg_detalhe.merge(metrics, on=[col_channel, col_segment], how='left')

    resultados = []
    resultados_detalhe = []
    outliers_clientes = []

    def venda_zero_counts_for(client_list):
        sub = sales_df[sales_df[col_client].isin(client_list)].copy()
        counts = {}
        for limite in [1, 2, 3]:
            months_to_check = [global_data_max - pd.DateOffset(months=i) for i in range(0, limite)]
            df_filtro = sub[sub['data'].isin(months_to_check) & (sub[col_valor] <= 0)]
            cont_por_cliente = df_filtro.groupby(col_client)['data'].nunique().reset_index(name='count_months')
            clientes_zerados = cont_por_cliente.loc[cont_por_cliente['count_months'] == limite, col_client].unique()
            counts[f'VendaZero_{limite*30}dias'] = int(len(clientes_zerados))
        return counts

    for delta in deltas:
        df_temp = df_agg.copy()
        df_temp['Limite_Superior'] = df_temp['Volume_Avg_PDV'] + delta * df_temp['Volume_Std_PDV']
        df_temp['Limite_Inferior'] = df_temp['Volume_Avg_PDV'] - delta * df_temp['Volume_Std_PDV']
        df_temp['outlier'] = (df_temp[col_valor] > df_temp['Limite_Superior']) | (df_temp[col_valor] < df_temp['Limite_Inferior'])

        resumo = df_temp[df_temp['outlier']].groupby([col_channel, col_segment]).agg(
            Qtd_Outliers=(col_client, 'count'),
            Limite_Superior=('Limite_Superior', 'first'),
            Limite_Inferior=('Limite_Inferior', 'first'),
            Volume_Total=('Volume_Total', 'first'),
            Volume_Avg_PDV=('Volume_Avg_PDV', 'first'),
            Volume_Std_PDV=('Volume_Std_PDV', 'first'),
            Volume_Outliers=(col_valor, 'sum')
        ).reset_index()
        resumo['Delta'] = delta
        resumo['VendaZero_30dias'] = 0
        resumo['VendaZero_60dias'] = 0
        resumo['VendaZero_90dias'] = 0

        for i, row in resumo.iterrows():
            ch, seg = row[col_channel], row[col_segment]
            clientes_outliers = df_temp.loc[
                (df_temp['outlier']) & (df_temp[col_channel] == ch) & (df_temp[col_segment] == seg),
                col_client
            ].unique()
            if len(clientes_outliers) > 0:
                counts = venda_zero_counts_for(clientes_outliers)
                resumo.at[i, 'VendaZero_30dias'] = counts['VendaZero_30dias']
                resumo.at[i, 'VendaZero_60dias'] = counts['VendaZero_60dias']
                resumo.at[i, 'VendaZero_90dias'] = counts['VendaZero_90dias']

        resultados.append(resumo)

        df_temp_det = df_agg_detalhe.copy()
        df_temp_det['Limite_Superior'] = df_temp_det['Volume_Avg_PDV'] + delta * df_temp_det['Volume_Std_PDV']
        df_temp_det['Limite_Inferior'] = df_temp_det['Volume_Avg_PDV'] - delta * df_temp_det['Volume_Std_PDV']
        df_temp_det['outlier'] = (df_temp_det[col_valor] > df_temp_det['Limite_Superior']) | (df_temp_det[col_valor] < df_temp_det['Limite_Inferior'])

        group_cols_det = [col_channel, col_segment]
        if col_ko_class in df_temp_det.columns:
            group_cols_det.append(col_ko_class)
        if col_sub_channel in df_temp_det.columns:
            group_cols_det.append(col_sub_channel)

        resumo_det = df_temp_det[df_temp_det['outlier']].groupby(group_cols_det).agg(
            Qtd_Outliers=(col_client, 'count'),
            Limite_Superior=('Limite_Superior', 'first'),
            Limite_Inferior=('Limite_Inferior', 'first'),
            Volume_Total=('Volume_Total', 'first'),
            Volume_Avg_PDV=('Volume_Avg_PDV', 'first'),
            Volume_Std_PDV=('Volume_Std_PDV', 'first'),
            Volume_Outliers=(col_valor, 'sum')
        ).reset_index()
        resumo_det['Delta'] = delta
        resumo_det['VendaZero_30dias'] = 0
        resumo_det['VendaZero_60dias'] = 0
        resumo_det['VendaZero_90dias'] = 0

        for i, row in resumo_det.iterrows():
            filtro = (df_temp_det['outlier']) & (df_temp_det[col_channel] == row[col_channel]) & (df_temp_det[col_segment] == row[col_segment])
            if col_ko_class in df_temp_det.columns:
                filtro &= (df_temp_det[col_ko_class] == row[col_ko_class])
            if col_sub_channel in df_temp_det.columns:
                filtro &= (df_temp_det[col_sub_channel] == row[col_sub_channel])

            clientes_outliers = df_temp_det.loc[filtro, col_client].unique()
            if len(clientes_outliers) > 0:
                counts = venda_zero_counts_for(clientes_outliers)
                resumo_det.at[i, 'VendaZero_30dias'] = counts['VendaZero_30dias']
                resumo_det.at[i, 'VendaZero_60dias'] = counts['VendaZero_60dias']
                resumo_det.at[i, 'VendaZero_90dias'] = counts['VendaZero_90dias']

        resultados_detalhe.append(resumo_det)

        clientes_out = df_temp[df_temp['outlier']].copy()
        clientes_out['Delta'] = delta
        clientes_out.rename(columns={col_valor: 'Volume_Total_Cliente'}, inplace=True)
        outliers_clientes.append(clientes_out)

    if resultados:
        df_resultados = pd.concat(resultados, ignore_index=True)
        df_resultados['Volume_AVG_Outlier'] = df_resultados.apply(
            lambda r: r['Volume_Outliers'] / r['Qtd_Outliers'] if r['Qtd_Outliers'] else 0, axis=1)
        df_resultados['Outlier_vs_Canal'] = df_resultados.apply(
            lambda r: r['Volume_AVG_Outlier'] / r['Volume_Avg_PDV'] - 1 if r['Volume_Avg_PDV'] else 0, axis=1)
    else:
        df_resultados = pd.DataFrame()

    if resultados_detalhe:
        df_resultado_detalhe = pd.concat(resultados_detalhe, ignore_index=True)
        df_resultado_detalhe['Volume_AVG_Outlier'] = df_resultado_detalhe.apply(
            lambda r: r['Volume_Outliers'] / r['Qtd_Outliers'] if r['Qtd_Outliers'] else 0, axis=1)
        df_resultado_detalhe['Outlier_vs_Canal'] = df_resultado_detalhe.apply(
            lambda r: r['Volume_AVG_Outlier'] / r['Volume_Avg_PDV'] - 1 if r['Volume_Avg_PDV'] else 0, axis=1)
    else:
        df_resultado_detalhe = pd.DataFrame()

    if outliers_clientes:
        df_outliers_clientes = pd.concat(outliers_clientes, ignore_index=True)
    else:
        df_outliers_clientes = pd.DataFrame()

    default_order_general = [
        'Delta', col_segment, col_channel,
        'Qtd_Outliers', 'Volume_Total', 'Volume_Avg_PDV', 'Volume_Std_PDV',
        'Volume_Outliers', 'Volume_AVG_Outlier', 'Outlier_vs_Canal',
        'Limite_Superior', 'Limite_Inferior',
        'VendaZero_30dias', 'VendaZero_60dias', 'VendaZero_90dias'
    ]
    if order_general is None:
        order_general = default_order_general

    if not df_resultados.empty:
        cols_existentes_gen = [c for c in order_general if c in df_resultados.columns]
        extra_cols_gen = [c for c in df_resultados.columns if c not in cols_existentes_gen]
        df_resultados = df_resultados[cols_existentes_gen + extra_cols_gen]

    default_order_detalhe = [
        'Delta', col_segment, col_ko_class, col_channel, col_sub_channel,
        'Qtd_Outliers', 'Volume_Total', 'Volume_Avg_PDV', 'Volume_Std_PDV',
        'Volume_Outliers', 'Volume_AVG_Outlier', 'Outlier_vs_Canal',
        'Limite_Superior', 'Limite_Inferior',
        'VendaZero_30dias', 'VendaZero_60dias', 'VendaZero_90dias'
    ]
    if order_detalhe is None:
        order_detalhe = default_order_detalhe

    if not df_resultado_detalhe.empty:
        cols_existentes_det = [c for c in order_detalhe if c in df_resultado_detalhe.columns]
        extra_cols_det = [c for c in df_resultado_detalhe.columns if c not in cols_existentes_det]
        df_resultado_detalhe = df_resultado_detalhe[cols_existentes_det + extra_cols_det]

    return df_resultados, df_resultado_detalhe, df_outliers_clientes


def venda_zero(df,limites=[1, 2, 3],col_client='Client_ID',col_mes='Mes',col_valor='Valor'):
    sales_df = df.copy()
    sales_df[col_mes] = sales_df[col_mes].astype(str).str.zfill(6)
    sales_df['ano'] = pd.to_datetime(sales_df[col_mes], format='%Y%m', errors='coerce').dt.year
    sales_df['mes_num'] = pd.to_datetime(sales_df[col_mes], format='%Y%m', errors='coerce').dt.month
    sales_df['data'] = pd.to_datetime(dict(year=sales_df['ano'], month=sales_df['mes_num'], day=1))
    data_max = sales_df['data'].max()

    resumo_dict = {}
    clientes_dict = {}
    total_clients = sales_df[col_client].nunique()

    for limite in limites:
        months_to_check = [data_max - pd.DateOffset(months=i) for i in range(0, limite)]
        df_filtro = sales_df[sales_df['data'].isin(months_to_check) & (sales_df[col_valor] <= 0)].copy()
        cont_por_cliente = (
            df_filtro
            .groupby(col_client)['data']
            .nunique()
            .reset_index(name='count_months')
        )
        clientes_selecionados = cont_por_cliente.loc[cont_por_cliente['count_months'] == limite, col_client].unique()
        resumo_dict[limite] = int(len(clientes_selecionados))
        clientes_dict[limite] = clientes_selecionados.tolist()

    resumo_dict['Total_Clientes'] = int(total_clients)
    resumo_clientes = pd.DataFrame([resumo_dict])
    resumo_clientes.columns = [col if col == 'Total_Clientes' else f"{int(col)*30} dias" for col in resumo_clientes.columns]
    clientes_por_limite = pd.DataFrame.from_dict(clientes_dict, orient='index').transpose()
    clientes_por_limite.columns = [f"{int(col)*30} dias" for col in clientes_por_limite.columns]

    return resumo_clientes, clientes_por_limite

# -----------------------------
# UI: Upload e execução
# -----------------------------

uploaded_file = st.file_uploader("Faça upload do CSV de entrada", type=["csv"], help="CSV com colunas: Client_ID, Mes (AAAAMM), Valor, Channel, Sub_Channel, etc.")

# parâmetros opcionais
# with st.expander("Parâmetros (opcionais)"):
#    incluir_venda_zero = st.checkbox("Incluir venda zero nas métricas de Resumo Canal", value=True)
#    deltas_input = st.text_input("Deltas para outlier (separados por vírgula)", value="1.5,2,2.5,3")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        st.stop()

    st.subheader("Preview dos dados (primeiras 5 linhas)")
    st.dataframe(df.head())

    # tentativa de normalizar nomes de colunas mais comuns
    rename_map = {
        'Client_id': 'Client_ID',
        'client_id': 'Client_ID',
        'Client Id': 'Client_ID',
        'Sales': 'Valor',
        'sales': 'Valor',
        'Subchannel': 'Sub_Channel',
        'SubChannel': 'Sub_Channel'
    }
    df = df.rename(columns=rename_map)

    if 'Client_ID' not in df.columns or 'Mes' not in df.columns or 'Valor' not in df.columns:
        st.error("O CSV precisa conter (pelo menos) as colunas: Client_ID, Mes, Valor. Verifique e reenvie.")
    else:
        run = st.button("▶️ Executar análise e gerar Excel")
        if run:
            # parse deltas
            try:
                deltas = [float(x.strip()) for x in deltas_input.split(',') if x.strip()]
            except:
                deltas = [1.5, 2, 2.5, 3]

            with st.spinner("Processando... (pode demorar dependendo do tamanho do arquivo)"):
                # aplica as funções
                try:
                    resumocanal = resumo_canal(df, incluir_venda_zero=incluir_venda_zero)
                except Exception as e:
                    st.error(f"Erro em resumo_canal: {e}")
                    resumocanal = pd.DataFrame()

                try:
                    resumosegmentacao = segmentacao_pareto_df(df)
                except Exception as e:
                    st.error(f"Erro em segmentacao_pareto_df: {e}")
                    resumosegmentacao = pd.DataFrame()

                try:
                    resumo_outlier, resumo_detalhe_outlier, resumo_outlier_cliente = detectar_outliers_multidelta(df, deltas=deltas)
                except Exception as e:
                    st.error(f"Erro em detectar_outliers_multidelta: {e}")
                    resumo_outlier = pd.DataFrame(); resumo_detalhe_outlier = pd.DataFrame(); resumo_outlier_cliente = pd.DataFrame()

                try:
                    resumo_vendazero, resumo_pdv_vendazero = venda_zero(df)
                except Exception as e:
                    st.error(f"Erro em venda_zero: {e}")
                    resumo_vendazero = pd.DataFrame(); resumo_pdv_vendazero = pd.DataFrame()

            st.success("Análises concluídas")

            # mostra resultados resumidos
            st.subheader("Resumo por Canal")
            st.dataframe(resumocanal)

            st.subheader("Segmentação Pareto")
            st.dataframe(resumosegmentacao)

            st.subheader("Outliers (resumo)")
            st.dataframe(resumo_outlier)

            st.subheader("Venda Zero (resumo)")
            st.dataframe(resumo_vendazero)

            # prepara arquivo excel para download
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                resumocanal.to_excel(writer, sheet_name="ResumoCanal", index=False)
                resumosegmentacao.to_excel(writer, sheet_name="SegmentacaoPareto", index=False)
                resumo_outlier.to_excel(writer, sheet_name="Outliers", index=False)
                resumo_detalhe_outlier.to_excel(writer, sheet_name="Detalhe_Outliers_KA", index=False)
                resumo_outlier_cliente.to_excel(writer, sheet_name="DetalheClientes_Outliers", index=False)
                resumo_vendazero.to_excel(writer, sheet_name="ResumoClientes_VendaZero", index=False)
                resumo_pdv_vendazero.to_excel(writer, sheet_name="DetalheClientes_VendaZero", index=False)


            output.seek(0)

            st.download_button(
                label="📥 Baixar Excel com resultados",
                data=output,
                file_name="resultado_tratado.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# -----------------------------
# Rodapé / instruções
# -----------------------------

st.markdown("---")
st.write("**Instruções rápidas:**")
st.write("1. Garanta que o CSV contenha todas as colunas necessárias: `Client_ID`, `Nome`, `Latitud`, `Longitud`, `Adress`, `Segment`, `KO_Class`, `Channel`, `Subchannel`, `Category_Product`, `'Mes' (formato AAAAMM)`, `Valor`, `")
st.write("2. Clique em `Executar análise`.")
st.write("3. Baixe o arquivo Excel gerado.")
