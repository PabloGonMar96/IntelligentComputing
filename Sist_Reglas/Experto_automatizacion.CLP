;Se pregunta el numero de procesos que se desea automatizar dentro del proyecto
(defrule n_procesos
	=>
	(printout t "¿Cuántos procesos desea automatizar?" crlf)
	(assert (numero (explode$ (readline))))
)

;El proyecto que se desea automatizar tiene mas de 5 procesos y menos de 10.
(defrule mayor_5
	(numero ?a)
	(test(>= ?a 5))
	(test(>= 10 ?a))
	=>
	(assert (proceso mediano))
)

;El proyecto que se desea automatizar tiene menos de 5 procesos.
(defrule menor_5
	(numero ?a)
	(test(> 5 ?a))
	=>
	(assert (proceso pequeno))
)


;El proyecto que se desea automatizar tiene mas de 10 procesos.
(defrule mayor_10
	(numero ?a)
	(test(> ?a 10))
	=>
	(assert (proceso grande))
)

(defrule comunicacion
	=>
	(printout t "¿Cómo es el proceso?(Combinacional/Secuencial)"crlf)
	(assert (comunicacion (explode$ (readline))))
)

(defrule comunicacion_distancia
	(or(comunicacion Combinacional)(comunicacion Secuencial))
	=>
	(printout t "Distancia entre CPU y proceso en metros: "crlf)
	(assert (com_distancia (explode$ (readline))))
)

(defrule periferia_desc
	(or(comunicacion Combinacional)(comunicacion Secuencial))
	(com_distancia ?y)
	(test (> ?y 100))
	=>
	(assert (periferia descentralizada))
	(assert (monitorizacion si))
)

(defrule periferia_mdesc
	(comunicacion Combinacional)
	(com_distancia ?y)
	(test (>= 100 ?y))
	(test (> ?y  50))
	=>
	(assert (periferia m_descentralizada))
	(assert (monitorizacion si))
)

(defrule periferia_cen
	(not(proceso grande))
	(comunicacion Secuencial)
	(com_distancia ?y)
	(test (>= 50 ?y))
	=>
	(assert (periferia centralizada))
	(assert (monitorizacion no))
)

(defrule periferia_cen1
	(proceso grande)
	(comunicacion Secuencial)
	(com_distancia ?y)
	(test (>= 50 ?y))
	=>
	(assert (periferia centralizada))
	(assert (monitorizacion si))
)

(defrule manejo_datos
	=>
	(printout t "¿Numero de DI/DO que se emplearán?")
	(assert (DI (explode$ (readline))))
)

(defrule digitalInput
	(DI ?b)
	(test (> ?b 10))
	=>
	(assert (modulo_DI extra))
)


;SALIDA DE LOS PROCESOS GRANDES

(defrule salida
	(or(periferia descentralizada)(periferia m_descentralizada))
	(modulo_DI extra)
	(or(proceso grande)(proceso mediano))
	(monitorizacion si)
	=>
	(printout t crlf)
	(printout t crlf)
	(printout t "Se deberia de implementar en su proyecto:"crlf)
	(printout t "- 1 PLC S7-1200 cuya CPU sea 1212AC/DC/Rly" crlf)
	(printout t "- 1 Módulo de periferia descentralizada ET1200S" crlf)
	(printout t "- 1 Módulo extra de 8DI/8DO" crlf)
	(printout t "- 1 repetidor de señal" crlf)
	(printout t "- 1 monitorizacion mediante un sistema SCADA "  crlf)
)

(defrule salida1
	(periferia centralizada)
	(proceso grande)
	(modulo_DI extra)
	(monitorizacion si)
	=>
	(printout t crlf)
	(printout t crlf)
	(printout t "Se deberia de implementar en su proyecto:"crlf)
	(printout t "- 1 PLC S7-1500 cuya CPU sea 1515AC/DC/Rly" crlf)
	(printout t "- 1 Módulo de periferia ET200S" crlf)
	(printout t "- 1 Módulo extra de 8DI/8DO" crlf)
)



;SALIDA DE LOS PROCESOS MEDIANOS

(defrule salida2
	(or(periferia descentralizada)(periferia m_descentralizada))
	(proceso mediano)
	(monitorizacion si)
	=>
	(printout t crlf)
	(printout t crlf)
	(printout t "Se deberia de implementar en su proyecto:"crlf)
	(printout t "- 1 PLC S7-1200 cuya CPU sea 1212AC/DC/Rly" crlf)
	(printout t "- 1 Módulo de periferia descentralizada ET400S" crlf)
	(printout t "- 1 repetidor de señal" crlf)
	(printout t "- 1 monitorizacion mediante un sistema MASTER "  crlf)
)

(defrule salida3
	(or(periferia descentralizada)(periferia m_descentralizada))
	(modulo_DI extra)
	(proceso mediano)
	(monitorizacion si)
	=>
	(printout t crlf)
	(printout t crlf)
	(printout t "Se deberia de implementar en su proyecto:"crlf)
	(printout t "- 1 PLC S7-1200 cuya CPU sea 1212DC/DC/PN" crlf)
	(printout t "- 1 Módulo de periferia ET200S" crlf)
	(printout t "- 1 Módulo extra de 8DI/8DO" crlf)
	(printout t "- 1 pantalla tactil HMI PolymathESA 6''"  crlf)
)

(defrule salida5
	(periferia centralizada)
	(modulo_DI extra)
	(proceso mediano)
	(monitorizacion no)
	=>
	(printout t crlf)
	(printout t crlf)
	(printout t "Se deberia de implementar en su proyecto:"crlf)
	(printout t "- 1 PLC S7-1200 cuya CPU sea 1212DC/DC/PN" crlf)
	(printout t "- 1 Módulo de periferia ET200S" crlf)
	(printout t "- 1 Módulo extra de 8DI/8DO" crlf)
)

(defrule salida6
	(periferia centralizada)
	(proceso mediano)
	(monitorizacion no)
	=>
	(printout t crlf)
	(printout t crlf)
	(printout t "Se deberia de implementar en su proyecto:"crlf)
	(printout t "- 1 PLC S7-1200 cuya CPU sea 1212DC/DC/PN" crlf)
	(printout t "- 1 Módulo de periferia ET200S" crlf)
)


; SALIDA PROCESO PEQUEÑO

(defrule salida7
	(comunicacion Secuencial)
	(not(or(proceso grande)(proceso mediano)))
	(monitorizacion no)
	(not(modulo_DI extra))
	=>
	(printout t crlf)
	(printout t crlf)
	(printout t "Se deberia de implementar en su proyecto:"crlf)
	(printout t "- 1 PLC S7-300 cuya CPU sea 315-2DP AC/DC/Rly" crlf)
	(printout t "- 1 Módulo de periferia ET200S" crlf)
)

(defrule salida8
	(comunicacion Combinacional)
	(proceso pequeno)
	(not(modulo_DI extra))
	=>
	(printout t crlf)
	(printout t crlf)
	(printout t "Se deberia de implementar en su proyecto:"crlf)
	(printout t "- 1 PLC S7-300 cuya CPU sea 315-2DP AC/DC/Rly" crlf)
	(printout t "- 1 Módulo de periferia ET200S" crlf)
)

