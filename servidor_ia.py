from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import unicodedata
import urllib.parse

app = Flask(__name__)
CORS(app)  # Permite que la interfaz se comunique con Flask sin bloqueos

nombre_archivo = "corpus.txt"
nombre_modelo = "modelo_pesos.pt"

if os.path.exists(nombre_archivo):
    with open(nombre_archivo, "r", encoding="utf-8", errors="ignore") as f:
        texto_corpus = f.read()
    chars = sorted(list(set(texto_corpus)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
else:
    chars = list("abcdefghijklmnopqrstuvwxyz0123456789 ,.!?")
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}


def codificar(s):
    return [stoi[c] for c in s if c in stoi]


def decodificar(l):
    return "".join([itos[i] for i in l])


def limpiar_texto(texto):
    nfkd_form = unicodedata.normalize("NFKD", texto)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).lower()


# Hiperparámetros corregidos para que coincidan con el entrenamiento:
block_size = 64     # Cambiado de 32 a 64
n_embd = 128        # Cambiado de 64 a 128
n_head = 4
n_layer = 4         # Cambiado de 3 o 2 a 4 (asegúrate de que sea el mismo número de capas)
dropout = 0.1
device = 'cpu'

def procesar_consulta(texto_usuario):
    texto = texto_usuario.lower().strip()
    
    # 1. Validación para consultas cortas o directas basadas en tu corpus
    if "forma de pago" in texto or "pagos" in texto or "métodos" in texto:
        return "Aceptamos pagos en efectivo, transferencias bancarias nacionales y pago móvil a la tasa oficial del BCV. 💳"
    
    if "inventario" in texto or "producto" in texto:
        return "Entra a Inventario, presiona el botón 'Nuevo Producto', llena el nombre, precio y cantidad, y guarda los cambios. 📦"
        
    if "hola" in texto:
        return "¡Bienvenido a Rojas Vconta Pro! 🤝 Somos tu aliado integral combinando la más alta tecnología administrativa y respaldo jurídico. ¿En qué podemos asesorarte hoy?"

    # 2. Si no es una palabra clave corta, pasa al modelo entrenado (modelo_pesos.pt)
    respuesta_modelo = generar_con_modelo_PyTorch(texto_usuario)
    return respuesta_modelo

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, None

    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=5):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = TransformerLanguageModel().to(device)
if os.path.exists(nombre_modelo):
    model.load_state_dict(
        torch.load(nombre_modelo, map_location=torch.device("cpu"))
    )
    model.eval()
    print("[*] Pesos del modelo de texto cargados en el servidor web.")

# Banco estructurado con preguntas frecuentes, soporte, sistema e información jurídica y tributaria
MAPEO_PREGUNTAS = {
    "cuáles son los datos para pago móvil": "Banco: Bancrecer, Teléfono: 0422-1487574, RIF: R-141023736.",
    "cuales son los datos para pago movil": "Banco: Bancrecer, Teléfono: 0422-1487574, RIF: R-141023736.",
    "pago movil": "Banco: Bancrecer, Teléfono: 0422-1487574, RIF: R-141023736.",
    "cuales son los metodos de pago disponibles": "Aceptamos pagos en efectivo, transferencias bancarias nacionales y pago móvil a la tasa oficial del BCV. 💳",
    "metodos de pago": "Aceptamos pagos en efectivo, transferencias bancarias nacionales y pago móvil a la tasa oficial del BCV. 💳",
    "como reporto un pago realizado": "Puedes reportar tu pago de forma directa ingresando a nuestra plataforma web oficial en https://vconta-pro.web.app, adjuntando el búncher o comprobante en la sección de contacto, o enviándolo a través de nuestro número de WhatsApp para procesar tu activación o entrega de inmediato. 📲",
    "reportar pago": "Puedes reportar tu pago de forma directa ingresando a nuestra plataforma web oficial en https://vconta-pro.web.app, adjuntando el búncher o comprobante en la sección de contacto, o enviándolo a través de nuestro número de WhatsApp para procesar tu activación o entrega de inmediato. 📲",
    "que incluye el sistema administrativo v-conta pro": "Incluye control de inventario A-Z, facturación multimoneda en dólares y bolívares, módulo POS y soporte técnico local. 🚀",
    "que incluye el sistema": "Incluye control de inventario A-Z, facturación multimoneda en dólares y bolívares, módulo POS y soporte técnico local. 🚀",
    "como puedo solicitar una demostracion del sistema": "Puedes escribirnos directamente a través de nuestro número de WhatsApp o hacer clic en el enlace de contacto en nuestra web. 📲",
    "demostracion": "Puedes escribirnos directamente a través de nuestro número de WhatsApp o hacer clic en el enlace de contacto en nuestra web. 📲",
    "el sistema funciona sin conexion a internet": "Sí, cuenta con una arquitectura local robusta en Windows con sincronización segura para garantizar operatividad continua. 💻",
    "sin conexion": "Sí, cuenta con una arquitectura local robusta en Windows con sincronización segura para garantizar operatividad continua. 💻",
    "ofrecen soporte tecnico y asesoria presencial": "Brindamos soporte técnico local y asesoría tecnológica e integral en Barcelona, Lechería y toda la zona oriental. 🛠️",
    "soporte tecnico": "Brindamos soporte técnico local y asesoría tecnológica e integral en Barcelona, Lechería y toda la zona oriental. 🛠️",
    "como calculan las tasas de cambio de divisas": "Las conversiones se realizan de forma automática y transparente utilizando la tasa oficial actualizada del Banco Central de Venezuela (BCV). 📈",
    "tasa de cambio": "Las conversiones se realizan de forma automática y transparente utilizando la tasa oficial actualizada del Banco Central de Venezuela (BCV). 📈",
    "que costo tiene la instalacion y configuracion inicial": "Contamos con tarifas especiales de lanzamiento e implementación inmediata desde la primera semana. Escríbenos para cotizar tu caso. 💵",
    "costo": "Contamos con tarifas especiales de lanzamiento e implementación inmediata desde la primera semana. Escríbenos para cotizar tu caso. 💵",
    "como se maneja el cierre de caja diario": "El software realiza cuadres automáticos y exactos al centavo para evitar diferencias inexplicables y agilizar el trabajo de tus cajeros. ⏱️",
    "cierre de caja": "El software realiza cuadres automáticos y exactos al centavo para evitar diferencias inexplicables y agilizar el trabajo de tus cajeros. ⏱️",
    "ofrecen servicios de asesoria legal y corporativa": "Sí, unimos la tecnología con el derecho mercantil, corporativo y penal para ofrecerte un respaldo jurídico inquebrantable. ⚖️",
    "asesoria legal": "Sí, unimos la tecnología con el derecho mercantil, corporativo y penal para ofrecerte un respaldo jurídico inquebrantable. ⚖️",
    "puedo supervisar varias sucursales desde un solo lugar": "Nuestra plataforma multisede te permite centralizar el control administrativo y el inventario de todas tus sucursales en tiempo real. 🌐",
    "varias sucursales": "Nuestra plataforma multisede te permite centralizar el control administrativo y el inventario de todas tus sucursales en tiempo real. 🌐",
    "como se vincula la asesoria juridica con el sistema v-conta pro": "Ofrecemos un respaldo legal corporativo integrado que garantiza que cada factura, contrato y movimiento administrativo cumpla estrictamente con la legislación vigente. ⚖️",
    "que aspectos del derecho mercantil cubre el servicio": "Cubrimos constitución de empresas, redacción de actas de asamblea, contratos comerciales, fusiones y blindaje jurídico corporativo para proteger tu patrimonio. 🏛️",
    "derecho mercantil": "Cubrimos constitución de empresas, redacción de actas de asamblea, contratos comerciales, fusiones y blindaje jurídico corporativo para proteger tu patrimonio. 🏛️",
    "el sistema cumple con las normativas fiscales y legales de facturacion": "Sí, V-Conta Pro está diseñado bajo los estándares legales y tributarios requeridos para asegurar la plena validez fiscal de tus operaciones comerciales. 📋",
    "normativas fiscales": "Sí, V-Conta Pro está diseñado bajo los estándares legales y tributarios requeridos para asegurar la plena validez fiscal de tus operaciones comerciales. 📋",
    "ofrecen defensa y asesoria en el area penal y laboral": "Contamos con litigio especializado en derecho laboral y penal corporativo, resguardando los intereses de tu empresa ante cualquier eventualidad judicial. 🛡️",
    "area penal y laboral": "Contamos con litigio especializado en derecho laboral y penal corporativo, resguardando los intereses de tu empresa ante cualquier eventualidad judicial. 🛡️",
    "como protegen los datos y la privacidad de mi empresa": "Implementamos una arquitectura local robusta y cifrada, garantizando que la información confidencial de tu negocio permanezca segura y bajo tu total control. 🔒",
    "privacidad": "Implementamos una arquitectura local robusta y cifrada, garantizando que la información confidencial de tu negocio permanezca segura y bajo tu total control. 🔒",
    "que ventajas tiene unir tecnologia y derecho en una sola asesoria": "Obtienes una solución integral donde la automatización tecnológica y la seguridad jurídica trabajan juntas para impulsar y blindar tu modelo de negocio. 🚀",
    "como puedo emitir una factura": "Ve al módulo POS o Facturación, ingresa los datos del cliente, selecciona los productos y haz clic en procesar pago. ¡Listo! 🧾",
    "emitir una factura": "Ve al módulo POS o Facturación, ingresa los datos del cliente, selecciona los productos y haz clic en procesar pago. ¡Listo! 🧾",
    "donde veo el reporte de ventas del dia": "En el menú principal, haz clic en Estadísticas o Reportes y selecciona la opción de ventas diarias. 📊",
    "reporte de ventas": "En el menú principal, haz clic en Estadísticas o Reportes y selecciona la opción de ventas diarias. 📊",
    "como agrego un nuevo producto al inventario": 'Entra a Inventario, presiona el botón "Nuevo Producto", llena el nombre, precio y cantidad, y guarda los cambios. 📦',
    "agregar nuevo producto": 'Entra a Inventario, presiona el botón "Nuevo Producto", llena el nombre, precio y cantidad, y guarda los cambios. 📦',
    "el sistema permite descuentos por producto": "Sí, puedes aplicar porcentajes o montos de descuento directamente desde la pantalla de cobro antes de cerrar la factura. 🏷️",
    "descuentos": "Sí, puedes aplicar porcentajes o montos de descuento directamente desde la pantalla de cobro antes de cerrar la factura. 🏷️",
    "como configuro la tasa del dolar del bcv": "La tasa se actualiza automáticamente al conectar a internet, o puedes ingresarla de forma manual en la configuración general. 💵",
    "tasa del dolar": "La tasa se actualiza automáticamente al conectar a internet, o puedes ingresarla de forma manual en la configuración general. 💵",
    "puedo buscar productos escaneando codigos de barras": "¡Claro que sí! Conecta tu lector de códigos de barras USB y el sistema agregará el artículo de inmediato. 📷",
    "codigo de barras": "¡Claro que sí! Conecta tu lector de códigos de barras USB y el sistema agregará el artículo de inmediato. 📷",
    "como hago un respaldo de la base de datos": 'Ve a Configuración > Sistema > Copia de Seguridad y haz clic en "Exportar Respaldo" para guardar tus datos a salvo. 💾',
    "respaldo de la base de datos": 'Ve a Configuración > Sistema > Copia de Seguridad y haz clic en "Exportar Respaldo" para guardar tus datos a salvo. 💾',
    "que hago si un cliente me paga en bolivares en efectivo": "Ingresas el monto en bolívares y el sistema calcula automáticamente el vuelto exacto en ambas monedas. 🧮",
    "pago en bolivares": "Ingresas el monto en bolívares y el sistema calcula automáticamente el vuelto exacto en ambas monedas. 🧮",
    "se pueden anular facturas emitidas por error": "Sí, con permisos de administrador puedes anular facturas y el inventario se reajustará de manera automática. 🔄",
    "anular facturas": "Sí, con permisos de administrador puedes anular facturas y el inventario se reajustará de manera automática. 🔄",
    "como recupero mi contraseña de acceso": 'Haz clic en "¿Olvidaste tu contraseña?" en la pantalla de inicio de sesión o pídele al administrador principal que la restablezca. 🔑',
    "recuperar contraseña": 'Haz clic en "¿Olvidaste tu contraseña?" en la pantalla de inicio de sesión o pídele al administrador principal que la restablezca. 🔑',
    "cuanto tiempo toma instalar el sistema": "La instalación y configuración básica toma menos de 30 minutos para que comiences a trabajar de inmediato. ⏱️",
    "tiempo de instalacion": "La instalación y configuración básica toma menos de 30 minutos para que comiences a trabajar de inmediato. ⏱️",
    "necesito impresoras especiales para los tickets": "El sistema es compatible con la gran mayoría de impresoras térmicas USB y de red estándar del mercado. 🖨️",
    "impresoras": "El sistema es compatible con la gran mayoría de impresoras térmicas USB y de red estándar del mercado. 🖨️",
    "puedo usar el sistema en varias computadoras": "Sí, puedes configurarlo en red local para conectar múltiples cajas y terminales al mismo servidor central. 🌐",
    "red local": "Sí, puedes configurarlo en red local para conectar múltiples cajas y terminales al mismo servidor central. 🌐",
    "como contacto al soporte tecnico": "Puedes comunicarte mediante nuestro número 04221487574, chat directo, WhatsApp o correo electrónico para asistencia inmediata. 📞",
    "contacto": "Puedes comunicarte mediante nuestro número 04221487574, chat directo, WhatsApp o correo electrónico para asistencia inmediata. 📞",
    "el sistema emite alertas de stock bajo": "Sí, te avisa en pantalla cuando un producto está por agotarse para que puedas reabastecer a tiempo. ⚠️",
    "stock bajo": "Sí, te avisa en pantalla cuando un producto está por agotarse para que puedas reabastecer a tiempo. ⚠️",
    "como hago para constituir mi empresa": "Te asesoramos y redactamos todos los documentos legales necesarios, incluyendo el acta constitutiva y estatutos sociales, cumpliendo con el SENIAT y el Registro Mercantil. 🏢",
    "constituir mi empresa": "Te asesoramos y redactamos todos los documentos legales necesarios, incluyendo el acta constitutiva y estatutos sociales, cumpliendo con el SENIAT y el Registro Mercantil. 🏢",
    "que necesito para registrar una compañia anonima (c.a.) o firma personal": "Necesitamos la reserva de denominación comercial, cédulas y RIF de los socios, la dirección fiscal y el capital social inicial para armar el expediente. 📋",
    "registrar compañia": "Necesitamos la reserva de denominación comercial, cédulas y RIF de los socios, la dirección fiscal y el capital social inicial para armar el expediente. 📋",
    "cuanto tiempo toma el proceso de constitucion de una empresa": "Una vez reunidos los recaudos y aprobada la reserva del nombre en el SAPI, el trámite ante el Registro Mercantil se completa de forma ágil y segura. ⏱️",
    "ofrecen asesoria para la obtencion del rif juridico y licencias municipales": "Sí, gestionamos tu inscripción en el SENIAT (RIF) y te asesoramos con los permisos municipales y bomberiles en la zona oriental. 🏛️",
    "rif y licencias": "Sí, gestionamos tu inscripción en el SENIAT (RIF) y te asesoramos con los permisos municipales y bomberiles en la zona oriental. 🏛️",
    "donde puedo ver o descargar el sistema v-conta pro": "Puedes acceder directamente a nuestra plataforma web oficial en https://vconta-pro.web.app para conocer todas las funciones y solicitar tu acceso. 🌐",
    "descargar el sistema": "Puedes acceder directamente a nuestra plataforma web oficial en https://vconta-pro.web.app para conocer todas las funciones y solicitar tu acceso. 🌐",
    "como calcula el sistema el iva en las facturas": "El sistema aplica automáticamente la alícuota del IVA vigente a los productos gravados, generando el desglose fiscal exacto en cada factura. 📊",
    "calcular el iva": "El sistema aplica automáticamente la alícuota del IVA vigente a los productos gravados, generando el desglose fiscal exacto en cada factura. 📊",
    "como se manejan las retenciones de igtf en pagos en divisas": "V-Conta Pro calcula de forma transparente y automática el Impuesto a las Grandes Transacciones Financieras (IGTF) cuando se paga en moneda extranjera. 💵",
    "retenciones de igtf": "V-Conta Pro calcula de forma transparente y automática el Impuesto a las Grandes Transacciones Financieras (IGTF) cuando se paga en moneda extranjera. 💵",
    "emiten reportes para las declaraciones fiscales ante el seniat": "Sí, genera libros de compra y venta listos y ordenados para facilitar tus declaraciones mensuales y de IVA ante el SENIAT. 📋",
    "reportes fiscales": "Sí, genera libros de compra y venta listos y ordenados para facilitar tus declaraciones mensuales y de IVA ante el SENIAT. 📋",
    "como se declaran y procesan los anticipos o retenciones de iva": "El sistema registra y totaliza los comprobantes de retención de clientes y proveedores para mantener tu contabilidad al día y sin errores. 🧮",
    "retenciones de iva": "El sistema registra y totaliza los comprobantes de retención de clientes y proveedores para mantener tu contabilidad al día y sin errores. 🧮",
    "como se declara el iva ante el seniat": "Para declarar el IVA debes ingresar al portal del SENIAT con tu usuario, seleccionar Procesos Tributarios > Declaración de IVA, cargar los libros de compras y ventas, verificar el monto resultante y confirmar la declaración para generar la planilla VEP y tu comprobante. 📋",
    "declarar el iva": "Para declarar el IVA debes ingresar al portal del SENIAT con tu usuario, seleccionar Procesos Tributarios > Declaración de IVA, cargar los libros de compras y ventas, verificar el monto resultante y confirmar la declaración para generar la planilla VEP y tu comprobante. 📋",
    "cuales son los pasos para declarar el impuesto al valor agregado": "1. Ingresar al portal oficial del SENIAT (Contribuyentes Ordinarios). 2. Ir a Procesos Tributarios > Declaración de IVA > Declaración y Pago. 3. Cargar las bases imponibles de ventas y compras junto con los créditos fiscales. 4. Validar el impuesto a pagar o el excedente. 5. Confirmar y descargar el comprobante y la planilla de pago VEP. 📄",
    "hola": "¡Bienvenido a Rojas Vconta Pro! 🤝 Somos tu aliado integral combinando la más alta tecnología administrativa (V-Conta Pro) y un sólido respaldo jurídico corporativo, mercantil y laboral en la zona oriental. ¿En qué podemos asesorarte hoy?",
    "libros contables": "Los libros contables deben llevarse cronológicamente, sin tachaduras, en idioma castellano y reflejando fielmente todas las operaciones. Su importancia radica en que garantizan el cumplimiento legal y fiscal ante el SENIAT y ofrecen información exacta para la gestión financiera y el blindaje jurídico de tu empresa. 📚",
    "como se deben llevar los libros contables": "Deben registrarse de forma cronológica, clara y ordenada, cumpliendo con las normativas del Código de Comercio y las leyes tributarias venezolanas, asegurando la integridad de los datos financieros. 📋",
    "cual es la importancia de los libros contables": "Son vitales porque proporcionan un historial financiero transparente, facilitan las auditorías, evitan fiscales y respaldan legalmente el patrimonio de la compañía. ⚖️",
    "cuales son los libros obligatorios": "Los libros obligatorios según el Código de Comercio son el Libro Diario, el Libro Mayor y el Libro de Inventarios, además de los libros de Compras y Ventas para efectos del IVA. 📊",
    "cuales son los deberes formales de los contribuyentes ante el seniat": "Los contribuyentes deben llevar los libros de contabilidad y registros especiales al día, emitir facturas y documentos equivalentes cumpliendo las providencias del SENIAT, presentar declaraciones oportunas y exhibir el RIF en un lugar visible. 📋",
    "deberes formales": "Los contribuyentes deben llevar los libros de contabilidad y registros especiales al día, emitir facturas y documentos equivalentes cumpliendo las providencias del SENIAT, presentar declaraciones oportunas y exhibir el RIF en un lugar visible. 📋",
    "como se sanciona el atraso en la declaracion de impuestos": "Según el Código Orgánico Tributario (COT), el retraso u omisión en la presentación de declaraciones se sanciona con multas pecuniarias calculadas en veces el tipo de cambio oficial de la moneda de mayor valor publicado por el BCV, además de los intereses moratorios correspondientes. ⚖️",
    "sanciona el atraso": "Según el Código Orgánico Tributario (COT), el retraso u omisión en la presentación de declaraciones se sanciona con multas pecuniarias calculadas en veces el tipo de cambio oficial de la moneda de mayor valor publicado por el BCV, además de los intereses moratorios correspondientes. ⚖️",
    "atraso en la declaracion": "Según el Código Orgánico Tributario (COT), el retraso u omisión en la presentación de declaraciones se sanciona con multas pecuniarias calculadas en veces el tipo de cambio oficial de la moneda de mayor valor publicado por el BCV, además de los intereses moratorios correspondientes. ⚖️",
    "que es un ilicito formal en materia tributaria": "Son todas aquellas acciones u omisiones violatorias de las normas de índole formal establecidas en el Código Orgánico Tributario o leyes especiales, tales como no emitir facturas, no llevar libros obligatorios o no presentar declaraciones en los plazos legales. 🏛️",
    "ilicito formal": "Son todas aquellas acciones u omisiones violatorias de las normas de índole formal establecidas en el Código Orgánico Tributario o leyes especiales, tales como no emitir facturas, no llevar libros obligatorios o no presentar declaraciones en los plazos legales. 🏛️",
    "ilícito formal": "Son todas aquellas acciones u omisiones violatorias de las normas de índole formal establecidas en el Código Orgánico Tributario o leyes especiales, tales como no emitir facturas, no llevar libros obligatorios o no presentar declaraciones en los plazos legales. 🏛️",
    "cual es la base legal de las retenciones de iva": "Se rigen por la Ley del Impuesto al Valor Agregado y sus providencias administrativas de agentes de retención, donde determinados contribuyentes (especiales) deben retener un porcentaje del IVA causado al momento del pago o abono en cuenta. 📊",
    "que plazos existen para el pago de los tributos nacionales": "Los plazos varían según el tipo de tributo y el calendario de sujetos pasivos especiales publicado anualmente por el SENIAT, generalmente estructurados por el último dígito del número de RIF. ⏱️",
    "plazos para el pago": "Los plazos varían según el tipo de tributo y el calendario de sujetos pasivos especiales publicado anualmente por el SENIAT, generalmente estructurados por el último dígito del número de RIF. ⏱️",
    "tributos nacionales": "Los plazos varían según el tipo de tributo y el calendario de sujetos pasivos especiales publicado anualmente por el SENIAT, generalmente estructurados por el último dígito del número de RIF. ⏱️",
    "multas del seniat": "Las multas en el derecho tributario venezolano se calculan y aplican principalmente en función del tipo de cambio oficial de la moneda de mayor valor publicado por el BCV. ⚖️",
}

# --- SECCIÓN DE ENRUTAMIENTO Y CONTROL DE PETICIONES ---


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "msg" not in data:
            return (
                jsonify({"status": "error", "message": "Falta el parámetro 'msg'"}),
                400,
            )

        mensaje_usuario = data["msg"]
        mensaje_limpio = limpiar_texto(mensaje_usuario).strip()

        # Detección inteligente para solicitudes de generación visual (Imágenes)
        palabras_clave_visuales = ["imagen", "dibujo", "genera", "crea", "haz", "fotografia", "ilustracion", "diseno", "foto", "logo"]
        if any(palabra in mensaje_limpio for palabra in palabras_clave_visuales):
            
            prompt_visual = mensaje_usuario.lower()
            for filler in [
                "genera una imagen de",
                "dame una imagen de",
                "crea un dibujo de",
                "generame una imagen de",
                "general una imagen de",
                "generame un a imagen de",
                "crea una foto de",
                "haz un diseño de",
                "fotografia de",
                "imagen de",
                "crea",
                "genera",
                "haz",
                "diseña",
                "disena",
            ]:
                prompt_visual = prompt_visual.replace(filler, "")
            
            prompt_visual = prompt_visual.strip()
            if not prompt_visual:
                prompt_visual = "arte abstracto corporativo y moderno"

            nfkd_form = unicodedata.normalize('NFKD', prompt_visual)
            prompt_limpio_ascii = nfkd_form.encode('ascii', 'ignore').decode('ascii')
            if not prompt_limpio_ascii.strip():
                prompt_limpio_ascii = "modern business technology concept"

            prompt_final = f"Professional high quality digital art and graphic design of {prompt_limpio_ascii}, 4k resolution, highly detailed"
            prompt_codificado = urllib.parse.quote(prompt_final)
            url_generada = f"https://image.pollinations.ai/prompt/{prompt_codificado}?width=1024&height=1024&nologo=true"

            return jsonify({"status": "success", "response": url_generada, "tipo": "imagen"})

        # --- EVALUACIÓN INTELIGENTE DE PREGUNTAS ---
        respuesta = None

        # 1. Coincidencia exacta limpia
        if mensaje_limpio in MAPEO_PREGUNTAS:
            respuesta = MAPEO_PREGUNTAS[mensaje_limpio]
        else:
            # 2. Búsqueda flexible mejorada: si el mensaje del usuario contiene palabras clave importantes
            for clave, rpta in MAPEO_PREGUNTAS.items():
                palabras_clave = [p for p in clave.split() if len(p) > 3]
                if palabras_clave and any(p in mensaje_limpio for p in palabras_clave):
                    # Si coincide con suficientes palabras clave o alguna clave fuerte (como "pago", "movil", "iva", etc.)
                    coincidencias = sum(1 for p in palabras_clave if p in mensaje_limpio)
                    if coincidencias >= max(1, len(palabras_clave) // 2):
                        respuesta = rpta
                        break
            
            # 3. Si no encuentra coincidencia parcial, usamos el modelo Transformer local entrenado
            # 3. Si no encuentra coincidencia parcial, usamos el modelo Transformer local entrenado pero totalmente limpio
            if not respuesta:
                try:
                    # Forzamos al modelo a iniciar el formato de respuesta para que responda ordenadamente
                    prompt_modelo = f"[PREGUNTA]: {mensaje_usuario}\n[RESPUESTA]:"
                    contexto_idx = torch.tensor(
                        [codificar(prompt_modelo)], dtype=torch.long, device=device
                    )
                    prediccion = model.generate(
                        contexto_idx, max_new_tokens=90, temperature=0.6
                    ).tolist()[0]
                    
                    generado = decodificar(prediccion)
                    
                    # Extraer únicamente lo que está después de [RESPUESTA]:
                    if "[RESPUESTA]:" in generado:
                        parte_respuesta = generado.split("[RESPUESTA]:")[-1]
                        # Cortar si el modelo intenta empezar otra pregunta o etiqueta
                        if "[PREGUNTA]:" in parte_respuesta:
                            parte_respuesta = parte_respuesta.split("[PREGUNTA]:")[0]
                        respuesta_limpia = parte_respuesta.strip()
                    else:
                        respuesta_limpia = generado.replace(prompt_modelo, "").strip()

                    # Validar que la respuesta tenga longitud útil y sentido
                    if len(respuesta_limpia) > 5 and not respuesta_limpia.startswith("["):
                        respuesta = respuesta_limpia
                    else:
                        respuesta = "Comprendo tu consulta. Para brindarte una asesoría contable o jurídica exacta sobre este tema, por favor comunícate directamente con nosotros a través de nuestro soporte o WhatsApp. 📲"
                except Exception:
                    respuesta = "Comprendo tu consulta. Para brindarte una asesoría contable o jurídica exacta sobre este tema, por favor comunícate directamente con nosotros a través de nuestro soporte o WhatsApp. 📲"

        return jsonify({"status": "success", "response": respuesta, "tipo": "texto"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            jsonify({"status": "error", "message": f"Error interno: {str(e)}"}),
            500,
        )


@app.route("/")
def index():
    with open("templates/index.html", "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)