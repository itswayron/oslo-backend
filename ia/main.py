import json
import os

import requests

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
app = FastAPI()
api_key = os.getenv("API_KEY_GEMINI")

if not api_key:
    raise Exception("API_KEY_GEMINI não encontrada no .env")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.5,
    convert_system_message_to_human=True
)


MANUAL_OPERACIONAL = """
DIRETRIZES OPERACIONAIS:

- Degelos periódicos ao longo do dia são esperados.
- Pequenos aumentos de temperatura durante degelo podem ser normais.
- Recuperação rápida após degelo indica funcionamento saudável.
- Oscilações leves e controladas são aceitáveis.

SINAIS DE ATENÇÃO:
- Recuperação acima de 20 minutos pode indicar perda de eficiência.
- Longos períodos fora da faixa ideal merecem acompanhamento.
- Oscilações térmicas muito frequentes indicam instabilidade.
- Temperatura elevada sem recuperação consistente pode indicar anomalia.
- Amplitude térmica muito alta pode indicar degelo agressivo ou falha operacional.

POSSÍVEIS CAUSAS:
- Abertura excessiva de portas.
- Sobrecarga de produtos.
- Problemas de vedação.
- Sensor inconsistente.
- Baixa eficiência do sistema.
- Degelo desregulado.

REGRAS:
- Não trate degelo automaticamente como falha.
- Não conclua defeitos mecânicos com certeza absoluta.
- Priorize manutenção preventiva.
"""


PROMPT_ANALISE = f"""
Você é um técnico virtual especializado em monitoramento de equipamentos de refrigeração comercial da Eletrofrio.

Sua função é interpretar dados técnicos processados pelo backend e responder clientes via WhatsApp.

O backend já realizou:
- análises estatísticas,
- detecção de eventos,
- cálculo de eficiência,
- identificação de anomalias.

Sua função NÃO é recalcular os dados.
Sua função é interpretar os dados de forma humana, técnica e útil.

{MANUAL_OPERACIONAL}

OBJETIVOS:
- Explicar comportamento do equipamento.
- Identificar possíveis riscos.
- Diferenciar comportamento normal de suspeito.
- Orientar ações preventivas.

REGRAS IMPORTANTES:
- Seja natural.
- Não use markdown.
- Não use listas.
- Não seja alarmista.
- Não invente informações.
- Não afirme falhas como certeza.
- Evite repetir números desnecessariamente.
- Mensagem curta.
- Entre 300 e 700 caracteres.
- Deve parecer conversa real de WhatsApp.
- Considere as preferências e instruções presentes na mensagem do cliente.
- Adapte o formato da resposta ao que foi solicitado.
"""


PROMPT_INTENCOES = """
Você é um classificador de intenções.

Sua função é converter mensagens em JSON válido.

INTENTS POSSÍVEIS:
- CONSULTAR_EQUIPAMENTO
- SAUDACAO
- AGRADECIMENTO
- DESCONHECIDO

REGRAS:
- Retorne APENAS JSON.
- Nunca explique.
- Nunca use markdown.

EXEMPLOS:

Usuário: Como está o freezer 12?
Resposta:
{
  "intent": "CONSULTAR_EQUIPAMENTO",
  "device_id": 12
}

Usuário: Bom dia
Resposta:
{
  "intent": "SAUDACAO"
}

Usuário: Obrigado
Resposta:
{
  "intent": "AGRADECIMENTO"
}
"""

def detectar_intencao(mensagem_usuario: str):
    mensagens = [
        SystemMessage(content=PROMPT_INTENCOES),
        HumanMessage(content=mensagem_usuario)
    ]
    resposta = llm.invoke(mensagens)
    try:
        dados = json.loads(resposta.content)
        return dados
    except Exception:
        return {
            "intent": "DESCONHECIDO"
        }

def gerar_analise(dados_equipamento: dict, message: str):
    mensagens = [
        SystemMessage(content=PROMPT_ANALISE),
        HumanMessage(content=f"""
            Mensagem do cliente: {message}
            Dados do equipamento: {dados_equipamento}

        Responda considerando exatamente o que o cliente pediu.
        """)
    ]
    resposta = llm.invoke(mensagens)
    return resposta.content

def buscar_dados_equipamento(device_id: int):
    response = requests.get(
        f"http://localhost:8080/oslo/analyze/{device_id}",
        timeout=10
    )
    response.raise_for_status()
    return response.json()


def obter_saudacao():
    from datetime import datetime
    hora = datetime.now().hour
    if hora < 12:
        return "Bom dia"
    if hora < 18:
        return "Boa tarde"

    return "Boa noite"

def montar_resposta_whatsapp(mensagem_usuario: str):
    resultado = detectar_intencao(mensagem_usuario)
    intent = resultado.get("intent")

    if intent == "SAUDACAO":
        return f"{obter_saudacao()}! Como posso ajudar com seus equipamentos hoje?"

    if intent == "AGRADECIMENTO":
        return "Por nada! Seguimos à disposição para acompanhar seus equipamentos."

    if intent == "CONSULTAR_EQUIPAMENTO":
        device_id = resultado.get("device_id")

        if not device_id:
            return "Não consegui identificar o equipamento solicitado."

        try:
            dados = buscar_dados_equipamento(device_id)
            analise = gerar_analise(dados, mensagem_usuario)
            return f"{analise}"

        except Exception:
            return "Não foi possível consultar os dados do equipamento neste momento."

    return "Não consegui entender sua solicitação. Você pode informar o número do equipamento?"

@app.post("/oslo/chat")
async def chat(payload: dict):
    try:
        mensagem_usuario = payload.get("mensagem")
        if not mensagem_usuario:
            return {
                "sucesso": False,
                "erro": "Mensagem não enviada"
            }

        resposta = montar_resposta_whatsapp(mensagem_usuario)
        return {
            "sucesso": True,
            "resposta": resposta
        }

    except Exception as e:
        return {
            "sucesso": False,
            "erro": str(e)
        }

@app.get("/")
async def healthcheck():
    return {
        "status": "online"
    }
