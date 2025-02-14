 Selección de las características y la variable target
X = df[['Temperatura_Media', 'Precipitaciones', 'Humedad_Relativa', 'Velocidad_Viento', 'Radiacion_Solar']]
y = df['Ventas']

# División del dataset en entrenamiento y prueba
st.subheader("División de los datos")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.write(f"Datos de entrenamiento: {X_train.shape}")
st.write(f"Datos de prueba: {X_test.shape}")

# Entrenamiento del modelo
st.subheader("Entrenamiento del modelo Gradient Boosting Regressor")
modelo_3 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
modelo_3.fit(X_train, y_train)

# Predicción y evaluación en datos de prueba
y_pred_modelo_3 = modelo_3.predict(X_test)
mae_modelo_3 = mean_absolute_error(y_test, y_pred_modelo_3)
rmse_modelo_3 = np.sqrt(mean_squared_error(y_test, y_pred_modelo_3))
r2_modelo_3 = r2_score(y_test, y_pred_modelo_3)

st.write("### Resultados del modelo")
st.write(f"**MAE:** {mae_modelo_3:.2f}")
st.write(f"**RMSE:** {rmse_modelo_3:.2f}")
st.write(f"**R²:** {r2_modelo_3:.2f}")

# Gráficos de predicciones
st.subheader("Gráficos de predicciones")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_modelo_3, color='blue')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
ax.set_xlabel('Ventas reales')
ax.set_ylabel('Ventas predichas')
ax.set_title('Ventas reales vs Ventas predichas')
st.pyplot(fig)

# Optimización de hiperparámetros con Grid Search
st.subheader("Optimización de hiperparámetros con Grid Search")
parametros_modelo_3 = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10] 
}

grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), parametros_modelo_3, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

st.write("**Mejores parámetros encontrados:**")
st.write(grid_search.best_params_)

# Mostrar el modelo final
best_model = grid_search.best_estimator_
st.write("### Mejor modelo basado en Grid Search")
st.write(best_model)
