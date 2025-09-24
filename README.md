# IML 1.2 - Análise de Aquisição de Produtos Bancários

## Descrição do Projeto

Este projeto implementa um sistema de classificação para prever a probabilidade de clientes subscreverem a um produto bancário (depósito a prazo). O objetivo é otimizar as campanhas de marketing identificando clientes com maior potencial de conversão.

## Arquivos do Projeto

- `bank_marketing_analysis.ipynb`: Notebook principal com análise completa
- `bank_marketing_functions.py`: Funções auxiliares para análise
- `Unidade 2 - Atividade2.csv`: Base de dados original
- `README.md`: Este arquivo de documentação

## Estrutura da Solução

### 1. Análise Exploratória dos Dados (EDA)
- Carregamento e inspeção inicial dos dados
- Análise da distribuição da variável alvo (desbalanceamento)
- Análise de variáveis categóricas e numéricas
- Identificação de valores missing e outliers

### 2. Tratamento e Preparação dos Dados
- Tratamento de valores 'unknown'
- Encoding de variáveis categóricas
- Normalização dos dados
- Divisão em conjuntos de treino e teste

### 3. Implementação de Modelos
- **Regressão Logística**: Baseline com class_weight='balanced'
- **Random Forest**: Ensemble com análise de importância de features
- **Gradient Boosting**: Algoritmo de boosting
- **Support Vector Machine**: SVM com kernel RBF

### 4. Avaliação e Comparação
- Métricas adequadas para dados desbalanceados:
  - AUC-ROC (capacidade geral de discriminação)
  - AUC-PR (precisão média)
  - F1-Score (equilíbrio entre precisão e recall)
  - Recall (capacidade de identificar casos positivos)
- Visualizações comparativas
- Seleção do melhor modelo

### 5. Sistema de Previsão
- Função para previsão de novos usuários
- Exemplo prático de uso
- Recomendações de negócio baseadas na probabilidade

## Como Executar

### Pré-requisitos
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Execução do Notebook
1. Abra o arquivo `bank_marketing_analysis.ipynb` no Jupyter Notebook
2. Execute todas as células sequencialmente
3. Os resultados serão exibidos automaticamente

### Uso das Funções Auxiliares
```python
from bank_marketing_functions import load_and_preprocess_data, predict_new_user

# Carregar e processar dados
df_processed, feature_columns, label_encoders = load_and_preprocess_data('Unidade 2 - Atividade2.csv')

# Fazer previsão para novo usuário
new_user = {
    'age': 35,
    'job': 'admin.',
    'marital': 'married',
    # ... outros campos
}

prediction, probability = predict_new_user(
    new_user, model, feature_columns, label_encoders
)
```

## Principais Descobertas

### Características dos Dados
- **Total**: 41.188 registros com 20 features
- **Desbalanceamento**: Apenas ~11% dos clientes subscreveram
- **Valores missing**: Presentes em algumas variáveis categóricas
- **Variáveis econômicas**: Indicadores macroeconômicos incluídos

### Tratamento Realizado
- Substituição de valores 'unknown' por valores conservadores
- Encoding de variáveis categóricas
- Normalização para algoritmos sensíveis à escala
- Manutenção de outliers (podem ser importantes)

### Critérios de Avaliação
Para dados desbalanceados, as métricas mais importantes são:
1. **AUC-ROC**: Capacidade geral de discriminação
2. **AUC-PR**: Precisão média (melhor para classes desbalanceadas)
3. **Recall**: Capacidade de identificar casos positivos
4. **F1-Score**: Equilíbrio entre precisão e recall

## Impactos de Negócio

### Benefícios Esperados
1. **Otimização de Recursos**: Foco nos clientes com maior potencial
2. **Redução de Custos**: Menos ligações desnecessárias
3. **Aumento da Conversão**: Melhor taxa de sucesso das campanhas
4. **Experiência do Cliente**: Menos incomodo para clientes não interessados
5. **Insights Estratégicos**: Compreensão dos padrões comportamentais

### Implementação Recomendada
1. **Teste A/B**: Implementação gradual com grupos de controle
2. **Monitoramento**: Acompanhamento de métricas de negócio
3. **Feedback Loop**: Coleta de feedback dos agentes de vendas
4. **Atualização Regular**: Retreinamento periódico do modelo

## Limitações e Melhorias Futuras

### Limitações Atuais
- Base altamente desbalanceada
- Dados históricos podem não refletir mudanças atuais
- Threshold fixo (0.5) pode não ser otimizado
- Variáveis econômicas podem estar desatualizadas

### Melhorias Sugeridas
1. **Técnicas de Balanceamento**: SMOTE, oversampling
2. **Feature Engineering**: Criação de novas variáveis
3. **Ensemble Methods**: Combinação de múltiplos modelos
4. **Threshold Optimization**: Ajuste baseado em custo-benefício
5. **Validação Temporal**: Simulação de cenários reais

## Conclusão

O projeto desenvolveu com sucesso um sistema de classificação para otimização de campanhas de marketing bancário. O modelo selecionado oferece uma base sólida para implementação, com potencial significativo de impacto positivo nos resultados de negócio.

A solução está pronta para implementação e pode ser facilmente integrada aos sistemas existentes do banco, proporcionando uma ferramenta valiosa para tomada de decisões baseadas em dados.
