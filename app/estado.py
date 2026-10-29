"""Pesos compartidos y consultas; los PNG recibidos viven solo durante la petición."""
from functools import lru_cache
from threading import RLock
from . import motor

_inferencia = RLock()

@lru_cache(maxsize=1)
def paquete_fusion():
    return motor.cargar_fusion()

@lru_cache(maxsize=1)
def paquete_imagen():
    fusion = paquete_fusion()
    return fusion.imagen if fusion is not None else motor.cargar_imagen()

@lru_cache(maxsize=1)
def ejemplos():
    return motor.ejemplos()

@lru_cache(maxsize=8)
def cortes_ejemplo(paciente):
    if paciente not in ejemplos().index:
        raise ValueError('La paciente seleccionada no está entre los ejemplos de entrenamiento.')
    return motor.cortes_de_ejemplo(paciente)

def prediccion(pixeles, nombres):
    with _inferencia:
        return motor.predecir_cortes(paquete_imagen(), pixeles, list(nombres))

def recorrido(corte):
    with _inferencia:
        return motor.recorrido(paquete_imagen().modelos[0], corte)

def visor(pixeles, nombres, p_cortes):
    return motor.visor(pixeles, list(nombres), list(p_cortes))

@lru_cache(maxsize=1)
def portada():
    return motor.portada()

@lru_cache(maxsize=1)
def capas_red():
    return motor.capas_red()

def bloques(corte):
    return motor.bloques(recorrido(corte), capas_red())
