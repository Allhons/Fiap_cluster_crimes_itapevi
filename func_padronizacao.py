import pandas as pd
import datetime
import unidecode

class Padronizacao:
    @staticmethod
    def ajustar_tags_descr_periodo(df):
        # Padroniza para maiúsculo e remove acentos
        df['DESCR_PERIODO'] = df['DESCR_PERIODO'].astype(str).apply(lambda x: unidecode.unidecode(x.strip().upper()))
        # Dicionário de conversão para padronizar as tags
        conversao = {
            "A NOITE": "De Noite",
            "DE NOITE": "De Noite",
            "DE MADRUGADA": "De Madrugada",
            "PELA MANHA": "De Manhã",
            "DE MANHA": "De Manhã",
            "A TARDE": "De Tarde",
            "DE TARDE": "De Tarde",
            "EM HORA INCERTA": "Em Hora Incerta",
            "": "Em Hora Incerta",         # Adiciona vazio explícito
            "NAN": "Em Hora Incerta",     # Adiciona string 'nan'
            "NAT": "Em Hora Incerta"      # Adiciona string 'NaT'
        }
        df['DESCR_PERIODO'] = df['DESCR_PERIODO'].replace(conversao)
        return df

    @staticmethod
    def periodo_por_hora(hora):
        if pd.isnull(hora):
            return None
        if isinstance(hora, datetime.datetime):
            hora = hora.time()
        elif isinstance(hora, str):
            try:
                hora = pd.to_datetime(hora, errors='coerce').time()
            except Exception:
                return None
        if hora >= datetime.time(0, 0, 0) and hora <= datetime.time(5, 59, 59):
            return "De Madrugada"
        elif hora >= datetime.time(6, 0, 0) and hora <= datetime.time(11, 59, 59):
            return "De Manhã"
        elif hora >= datetime.time(12, 0, 0) and hora <= datetime.time(17, 59, 59):
            return "De Tarde"
        elif hora >= datetime.time(18, 0, 0) and hora <= datetime.time(23, 59, 59):
            return "De Noite"
        else:
            return None

    @staticmethod
    def imputar_hora_por_natureza(df):
        # Para cada natureza e tag, calcula a hora mais comum
        referencia = (
            df.dropna(subset=['HORA_OCORRENCIA_BO'])
              .groupby(['NATUREZA_APURADA', 'DESCR_PERIODO'])['HORA_OCORRENCIA_BO']
              .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
              .reset_index()
        )
        # Função para imputar hora nas linhas nulas
        def imputar(row):
            if pd.isnull(row['HORA_OCORRENCIA_BO']):
                filtro = (
                    (referencia['NATUREZA_APURADA'] == row['NATUREZA_APURADA']) &
                    (referencia['DESCR_PERIODO'] == row['DESCR_PERIODO'])
                )
                resultado = referencia.loc[filtro, 'HORA_OCORRENCIA_BO']
                if not resultado.empty:
                    return resultado.iloc[0]
            return row['HORA_OCORRENCIA_BO']
        df['HORA_OCORRENCIA_BO'] = df.apply(imputar, axis=1)
        return df

    @staticmethod
    def taguear_periodo(df):
        # Remove espaços e converte vazios explícitos para NaN
        df['HORA_OCORRENCIA_BO'] = df['HORA_OCORRENCIA_BO'].astype(str).str.strip().replace({'': None, 'nan': None, 'NaT': None})

        # 1. Padroniza as tags da coluna DESCR_PERIODO
        df = Padronizacao.ajustar_tags_descr_periodo(df)

        # 2. Ajusta as tags de acordo com a hora, se houver hora
        df['DESCR_PERIODO'] = df.apply(
            lambda row: Padronizacao.periodo_por_hora(row['HORA_OCORRENCIA_BO']) 
            if pd.notnull(row['HORA_OCORRENCIA_BO']) else row['DESCR_PERIODO'], axis=1
        )

        # 3. Imputa hora nas linhas nulas usando a natureza e tag (ainda como string)
        df = Padronizacao.imputar_hora_por_natureza(df)

        # 4. Agora sim, converte para datetime.time
        df['HORA_OCORRENCIA_BO'] = pd.to_datetime(
            df['HORA_OCORRENCIA_BO'], errors='coerce'
        ).dt.time

        # 5. Após imputar, ajusta novamente as tags para garantir consistência
        df['DESCR_PERIODO'] = df.apply(
            lambda row: Padronizacao.periodo_por_hora(row['HORA_OCORRENCIA_BO']) 
            if pd.notnull(row['HORA_OCORRENCIA_BO']) else row['DESCR_PERIODO'], axis=1
        )

        # 6. Se ainda houver nulo, marca como "Em Hora Incerta"
        df['DESCR_PERIODO'] = df['DESCR_PERIODO'].fillna("Em Hora Incerta")
        
        # Print do motivo dos nulos
        for idx, row in df[df['HORA_OCORRENCIA_BO'].isnull()].iterrows():
            if row['DESCR_PERIODO'] == "Em Hora Incerta":
                print(f"Linha {idx}: Sem hora e sem tag válida.")
            else:
                print(f"Linha {idx}: Sem hora disponível para NATUREZA_APURADA={row['NATUREZA_APURADA']} e DESCR_PERIODO={row['DESCR_PERIODO']}.")

        return df
    @staticmethod
    def Excluindo_dados_de_divulgacao_vedados(df):
        # Remove linhas onde o valor da coluna LOGRADOURO é exatamente "VEDAÇÃO DA DIVULGAÇÃO DOS DADOS RELATIVOS"
        df = df[df['LOGRADOURO'] != "VEDAÇÃO DA DIVULGAÇÃO DOS DADOS RELATIVOS"]
        return df

    @staticmethod
    def ajustando_latitude_longitude(df):
        import pandas as pd
        import numpy as np
        """
        Ajusta valores de LATITUDE e LONGITUDE quando ausentes (0, NULL, NaN ou vazio).
        
        Lógica:
        1. Identifica linhas com LATITUDE inválida.
        2. Pega o LOGRADOURO e o NUMERO da linha inválida.
        3. Filtra o DataFrame para a mesma rua.
        4. Busca o número mais próximo.
        5. Copia LATITUDE e LONGITUDE da linha encontrada para a linha original.
        """

        # Garantir consistência nos dados
        df = df.copy()

        # Normalizar LATITUDE e LONGITUDE
        for col in ["LATITUDE", "LONGITUDE"]:
            df[col] = df[col].replace(["NULL", "NUL,L", "", " "], np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Iterar sobre linhas com LATITUDE ou LONGITUDE inválidas
        condicao_invalida = (
            df["LATITUDE"].isna() | (df["LATITUDE"] == 0) |
            df["LONGITUDE"].isna() | (df["LONGITUDE"] == 0)
        )

        for idx, row in df[condicao_invalida].iterrows():
            logradouro = row["LOGRADOURO"]
            numero_ref = row["NUMERO_LOGRADOURO"]

            # Filtra mesma rua com coordenadas válidas
            candidatos = df[
                (df["LOGRADOURO"] == logradouro) &
                (~df["LATITUDE"].isna()) & (df["LATITUDE"] != 0) &
                (~df["LONGITUDE"].isna()) & (df["LONGITUDE"] != 0)
            ].copy()

            if not candidatos.empty:
                # Busca número mais próximo
                candidatos["distancia_numero"] = (candidatos["NUMERO_LOGRADOURO"] - numero_ref).abs()
                melhor = candidatos.sort_values("distancia_numero").iloc[0]

                # Copia LATITUDE/LONGITUDE
                df.at[idx, "LATITUDE"] = melhor["LATITUDE"]
                df.at[idx, "LONGITUDE"] = melhor["LONGITUDE"]

        return df
    
    @staticmethod
    def padronizar_logradouros(df):
        # Remove espaços extras e coloca cada palavra com inicial maiúscula
        df['LOGRADOURO'] = df['LOGRADOURO'].astype(str).str.strip().str.title()
        return df
    
    def preencher_lat_long_vazias(df, cidade="Itapevi", estado="SP", pais="Brasil"):
        import pandas as pd
        import numpy as np
        from geopy.geocoders import Nominatim
        from geopy.extra.rate_limiter import RateLimiter
        
        """
        Percorre o DataFrame e completa LATITUDE/LONGITUDE usando geocodificação
        sempre que encontrar valores nulos ou iguais a 0.
        """

        df = df.copy()

        # Normalizar valores inválidos (0, vazio, NULL etc.)
        for col in ["LATITUDE", "LONGITUDE"]:
            df[col] = df[col].replace(["NULL", "NUL,L", "", " "], np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Configurar Nominatim (OpenStreetMap)
        geolocator = Nominatim(user_agent="ajuste_latlong")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)  # 1 req/s

        # Percorrer linhas com lat/long inválidas
        condicao_invalida = (
            df["LATITUDE"].isna() | (df["LATITUDE"] == 0) |
            df["LONGITUDE"].isna() | (df["LONGITUDE"] == 0)
        )

        for idx, row in df[condicao_invalida].iterrows():
            logradouro = str(row["LOGRADOURO"])
            numero = str(row["NUMERO_LOGRADOURO"])

            endereco = f"{logradouro}, {numero}, {cidade}, {estado}, {pais}"

            try:
                location = geocode(endereco)
                if location:
                    df.at[idx, "LATITUDE"] = location.latitude
                    df.at[idx, "LONGITUDE"] = location.longitude
                    print(f"[OK] {endereco} -> {location.latitude}, {location.longitude}")
                    
                else:
                    print(f"[ERRO] Não encontrado: {endereco}")
            except Exception as e:
                print(f"[ERRO] {endereco}: {e}")

        return df