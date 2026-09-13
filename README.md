# Markowitz_Wholesalesale

Dashboard desktop avançado que combina **Machine Learning** e **Computação Quântica** para análise de portfólios financeiros, baseado na teoria de Markowitz (otimização média-variância).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-aqua.svg)

---

## Visão Geral

Este projeto é uma aplicação desktop profissional que integra:

- **Machine Learning** - Previsão de séries temporais com Regressão Linear e Random Forest
- **Computação Quântica** - Otimização de portfólio usando QAOA (Quantum Approximate Optimization Algorithm)
- **Análise Estatística** - Visualizações avançadas de dados financeiros
- **Interface Moderna** - UI/UX profissional com CustomTkinter

---

## Funcionalidades

### Previsão com Machine Learning
- Modelos: **Regressão Linear** e **Random Forest**
- Horizonte de previsão ajustável (1-30 períodos)
- Visualização comparativa: dados reais vs. previsão
- Suporte a múltiplas colunas simultâneas

### Otimização Quântica (QAOA)
- Implementação do algoritmo **Markowitz** via computação quântica
- Seleção binária de ativos (comprar/não comprar)
- Balanceamento automático entre retorno e risco
- Restrições de orçamento configuráveis

### Análise Estatística
- **Boxplot** - Distribuição dos dados
- **Histograma** - Frequência de valores
- **Matriz de Correlação** - Relações entre ativos
- **Retornos** - Análise de variações percentuais

### Interface de Usuário
- Tema **verde aqua** customizado
- Modo escuro nativo
- Logs em tempo real
- Feedback visual de todas as operações
- Sidebar com controles intuitivos

---

## 📋 Requisitos

### Dependências Python

```bash

pip install customtkinter pandas numpy scikit-learn matplotlib qiskit qiskit-optimization qiskit-algorithms yfinance

```
 ®


