import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.title("Análise de Padrões de Compras em Supermercado")
st.write("""
    Este projeto analisa padrões de compras em um supermercado, utilizando técnicas de machine learning para agrupar clientes com comportamentos de compra semelhantes. 
    Estes grupos, ou "clusters", permitem compreender melhor os hábitos dos clientes e identificar potenciais promoções combinadas.
""")

df = pd.read_csv('Groceries_dataset.csv')
st.write("Primeiras linhas dos dados:")
st.write(df.head())

df['itemDescription'] = df['itemDescription'].replace({
    'whole milk': 'leite integral', 'other vegetables': 'outros vegetais', 'rolls/buns': 'pães',
    'soda': 'refrigerante', 'yogurt': 'iogurte', 'root vegetables': 'vegetais de raiz', 'tropical fruit': 'fruta tropical',
    'bottled water': 'água engarrafada', 'sausage': 'salsicha', 'citrus fruit': 'fruta cítrica'
})

customer_basket = df.groupby('Member_number')['itemDescription'].apply(list).reset_index()
customer_basket['num_items'] = customer_basket['itemDescription'].apply(lambda x: len(set(x)))
customer_basket['total_items'] = customer_basket['itemDescription'].apply(lambda x: len(x))

st.subheader("Distribuição de Número de Itens Comprados por Cliente")
st.write("Aqui vemos a quantidade de itens diferentes comprados por cliente. Isso nos ajuda a entender a variedade de produtos consumidos em cada compra.")
fig, ax = plt.subplots()
sns.histplot(customer_basket['num_items'], bins=30, kde=True, ax=ax)
st.pyplot(fig)

st.subheader("O que são Clusters?")
st.write("""
    Clusters são grupos de clientes com comportamentos semelhantes. Usando o algoritmo K-means, conseguimos segmentar os clientes com base em quantos itens diferentes eles compram e quantos itens no total estão no seu carrinho.
    Isso ajuda a identificar padrões e comportamentos de compra, como clientes que compram muitos itens variados, em comparação com aqueles que compram menos tipos, mas em maior quantidade.
""")

X = customer_basket[['num_items', 'total_items']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=0)
customer_basket['cluster'] = kmeans.fit_predict(X_scaled)

st.subheader("Segmentação de Clientes em Clusters")
fig, ax = plt.subplots()
sns.countplot(x='cluster', data=customer_basket, ax=ax)
st.write("Cada cluster representa um grupo de clientes com comportamento de compra semelhante.")
st.pyplot(fig)

st.write("Características dos Clusters:")
st.write("""
    Abaixo, vemos as características médias dos clusters, mostrando o número médio de tipos de itens e o total de itens comprados em cada grupo.
    Isso pode ajudar a identificar potenciais grupos de clientes para promoções específicas.
""")
st.write(customer_basket.groupby('cluster').agg({
    'num_items': 'mean',
    'total_items': 'mean'
}).reset_index())

st.subheader("Itens Mais Comprados por Cluster")
for i in range(3):
    st.write(f"Cluster {i}:")
    items = customer_basket[customer_basket['cluster'] == i]['itemDescription'].explode().value_counts().head(10)
    st.write(items)

st.subheader("Análise de Itens Mais Comprados e Promoções")
st.write("Abaixo, temos gráficos dos itens e pares de itens mais comprados, sugerindo possíveis promoções de combo.")

item_counts = df['itemDescription'].value_counts().head(10)
fig, ax = plt.subplots()
item_counts.plot(kind='bar', ax=ax)
ax.set_title("Top 10 Itens Mais Comprados")
st.pyplot(fig)

basket = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).reset_index()
encoder = TransactionEncoder()
onehot = encoder.fit(basket['itemDescription']).transform(basket['itemDescription'])
basket_encoded = pd.DataFrame(onehot, columns=encoder.columns_)
frequent_items = apriori(basket_encoded, min_support=0.005, use_colnames=True)
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.3)

st.write("Pares de Itens Mais Comprados (Sugestão de Promoções):")
st.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))



