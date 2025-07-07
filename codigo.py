import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob, os, random


# ---------- helpers ----------
def cargar_datos(path_healthy, path_parkinson, max_imgs=300, tam=(64, 64)):
    healthy = glob.glob(os.path.join(path_healthy, "*.png"))[:max_imgs]
    park    = glob.glob(os.path.join(path_parkinson, "*.png"))[:max_imgs]

    X, y = [], []
    for f in healthy:
        img = Image.open(f).convert("L").resize(tam)
        X.append(np.asarray(img, dtype=np.float32).flatten())
        y.append(0)
    for f in park:
        img = Image.open(f).convert("L").resize(tam)
        X.append(np.asarray(img, dtype=np.float32).flatten())
        y.append(1)

    print(f"Total imágenes: {len(X)}")
    return np.array(X), np.array(y)

def dividir_datos(X, y, test_ratio=0.2):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    n_test = int(len(idx) * test_ratio)
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]


#------------------------------------------------DESCENSO POR GRADIENTE------------------------------------------------

# ---------- función f ------------------------------------
def f_wb(x, w, b): 
    z = np.dot(w, x) + b 
    return (np.tanh(z) + 1) / 2

# ---------- pérdida y exactitud ----------------------------
def loss(w, b, X, D): 
    preds = np.array([f_wb(xi, w, b) for xi in X]) 
    return np.mean((preds - D) ** 2) 

def accuracy(w, b, X, D): 
    preds = np.array([f_wb(xi, w, b) for xi in X]) 
    preds_bin = (preds >= 0.5).astype(int) 
    return np.mean(preds_bin == D) 

# ---------- gradientes -------------------------------------
def grad_L(w, b, X, D):
    z = X @ w + b                # vector (N,)
    tanh_z   = np.tanh(z)
    f   = 0.5 * (1 + tanh_z)
    common_term   = (1 - tanh_z**2) * (f - D)
    grad_w = (common_term[:, None] * X).mean(axis=0)  
    grad_b = common_term.mean()                       

    return grad_w, grad_b

# ---------- predicción y matriz de confusión ----------
def predecir_etiquetas(X, w, b):
    """
    Predice etiquetas usando los parámetros w y b entrenados
    """
    preds = np.array([f_wb(xi, w, b) for xi in X])
    preds_bin = (preds >= 0.5).astype(int)
    return preds_bin, preds

def generar_matriz_confusion(y_true, y_pred):

    # Calcular matriz de confusión manualmente
    tn = fp = fn = tp = 0
    for true, pred in zip(y_true, y_pred):
        if true == 0 and pred == 0:  # True Negative
            tn += 1
        elif true == 0 and pred == 1:  # False Positive
            fp += 1
        elif true == 1 and pred == 0:  # False Negative
            fn += 1
        elif true == 1 and pred == 1:  # True Positive
            tp += 1
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    # Calcular métricas
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # Precision, Recall, F1 para clase positiva (Parkinson)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calcular métricas para clase Healthy (negativa)
    precision_healthy = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_healthy = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_healthy = 2 * (precision_healthy * recall_healthy) / (precision_healthy + recall_healthy) if (precision_healthy + recall_healthy) > 0 else 0
    
    # Crear reporte manual
    report = f"""
Reporte de Clasificación:
              precision    recall  f1-score   support

Healthy          {precision_healthy:.3f}      {recall_healthy:.3f}      {f1_healthy:.3f}        {tn+fp}
Parkinson        {precision:.3f}      {recall:.3f}      {f1:.3f}       {tp+fn}

accuracy                            {accuracy:.3f}      {len(y_true)}
macro avg        {(precision_healthy + precision)/2:.3f}      {(recall_healthy + recall)/2:.3f}      {(f1_healthy + f1)/2:.3f}       {len(y_true)}
weighted avg     {(precision_healthy + precision)/2:.3f}      {(recall_healthy + recall)/2:.3f}      {(f1_healthy + f1)/2:.3f}       {len(y_true)}
"""
    
    return cm, accuracy, report

def graficar_matriz_confusion(cm, accuracy, title="Matriz de Confusión"):

    plt.figure(figsize=(8, 6))
    
    # Crear heatmap manualmente
    im = plt.imshow(cm, cmap='Blues', aspect='auto')
    
    # Agregar texto en cada celda
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', 
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)
    
    # Configurar ejes
    plt.xticks([0, 1], ['Healthy', 'Parkinson'])
    plt.yticks([0, 1], ['Healthy', 'Parkinson'])
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.title(f'{title}\nAccuracy: {accuracy:.4f}')
    
    # Agregar barra de color
    plt.colorbar(im)
    plt.show()

def analizar_convergencia_alpha(X_train, D_train, X_test, D_test, alphas=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2], iterations=1000):
    """
    Analiza la convergencia del método para diferentes valores de α
    """
    resultados_convergencia = {}
    
    print("Analizando convergencia para diferentes valores de α...")
    print("="*60)
    
    for alpha in alphas:
        print(f"\nEntrenando con α = {alpha}")
        
        # Fijar semilla para reproducibilidad
        np.random.seed(42)
        
        # Entrenar modelo y obtener historial completo
        w, b, loss_hist, acc_tr_hist, loss_te_hist, acc_te_hist = gradient_descent(X_train, D_train, X_test, D_test, alpha, iterations)
        
        resultados_convergencia[alpha] = {
            'loss_hist': loss_hist,
            'acc_tr_hist': acc_tr_hist,
            'acc_te_hist': acc_te_hist,
            'loss_final': loss_hist[-1],
            'acc_tr_final': acc_tr_hist[-1],
            'acc_te_final': acc_te_hist[-1]
        }
        
        print(f"  Loss final: {loss_hist[-1]:.6f}")
        print(f"  Accuracy Train final: {acc_tr_hist[-1]:.4f}")
        print(f"  Accuracy Test final: {acc_te_hist[-1]:.4f}")
    
    return resultados_convergencia

def graficar_convergencia_alpha(resultados_convergencia):
    """
    Grafica los resultados de convergencia para diferentes valores de α
    """
    alphas = list(resultados_convergencia.keys())
    colors = ['blue', '#0077b6', '#023e8a', '#0096c7', '#48cae4']
    
    # Gráfico 1: Convergencia de Accuracy de Test para diferentes α
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    for i, alpha in enumerate(alphas):
        acc_te_hist = resultados_convergencia[alpha]['acc_te_hist']
        plt.plot(acc_te_hist, color=colors[i], linewidth=2, label=f'α={alpha}')
    
    plt.title('Convergencia de Accuracy de Testing vs α')
    plt.xlabel('Iteraciones (cada 10)')
    plt.ylabel('Accuracy de Testing')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 2: Convergencia de Loss de Test para diferentes α
    plt.subplot(1, 2, 2)
    for i, alpha in enumerate(alphas):
        loss_hist = resultados_convergencia[alpha]['loss_hist']
        plt.plot(loss_hist, color=colors[i], linewidth=2, label=f'α={alpha}')
    
    plt.title('Convergencia de Loss de Testing vs α')
    plt.xlabel('Iteraciones (cada 10)')
    plt.ylabel('Loss de Testing')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Tabla resumen de convergencia
    print("\n" + "="*80)
    print("RESUMEN DE CONVERGENCIA DEL CONJUNTO DE TESTING")
    print("="*80)
    print(f"{'α':<10} {'Loss Test':<12} {'Acc Test':<12} {'Converge':<10}")
    print("-" * 80)
    
    for alpha in alphas:
        res = resultados_convergencia[alpha]
        acc_te_hist = res['acc_te_hist']
        
        # Verificar si converge (accuracy aumenta y se estabiliza)
        acc_inicial = acc_te_hist[0]
        acc_final = acc_te_hist[-1]
        acc_penultimo = acc_te_hist[-2] if len(acc_te_hist) > 1 else acc_final
        
        # Criterio de convergencia: accuracy aumenta y se estabiliza
        converge = "Sí" if (acc_final > acc_inicial and abs(acc_final - acc_penultimo) < 1e-4) else "No"
        
        print(f"{alpha:<10} {res['loss_final']:<12.6f} {res['acc_te_final']:<12.4f} {converge:<10}")
    
    print("="*80)

# ---------- análisis de diferentes tamaños de imagen ----------
def analizar_tamanos_imagen(path_healthy, path_parkinson, tamanos=[16, 32, 64, 128], max_imgs=300):
    """
    Analiza el impacto de diferentes tamaños de imagen en el rendimiento
    """
    import time
    resultados = {}
    
    for tam in tamanos:
        print(f"\n{'='*50}")
        print(f"ANALIZANDO TAMAÑO {tam}x{tam}")
        print(f"{'='*50}")
        
        # Cargar datos con este tamaño
        X, y = cargar_datos(path_healthy, path_parkinson, max_imgs=max_imgs, tam=(tam, tam))
        X_tr, X_te, y_tr, y_te = dividir_datos(X, y)
        
        # Normalizar datos
        X_tr_norm = X_tr / 255.0
        X_te_norm = X_te / 255.0
        
        print(f"Dimensiones: {X_tr.shape[1]} características")
        print(f"Tamaño del vector: {tam*tam} elementos")
        
        # Medir tiempo de entrenamiento
        inicio = time.time()
        
        # Entrenar modelo
        K = X_tr_norm.shape[1] 
        w = np.random.randn(K) * 0.01 
        b = 0.0 
        
        history = { 
            'train_loss': [], 
            'train_acc': [], 
            'test_loss': [], 
            'test_acc': [] 
        } 
        
        for it in range(1000): 
            grad_w, grad_b = grad_L(w, b, X_tr_norm, y_tr) 
            w -= 1e-5 * grad_w 
            b -= 1e-5 * grad_b 
            
            if it % 100 == 0 or it == 999: 
                train_loss = loss(w, b, X_tr_norm, y_tr) 
                train_acc = accuracy(w, b, X_tr_norm, y_tr) 
                test_loss = loss(w, b, X_te_norm, y_te) 
                test_acc = accuracy(w, b, X_te_norm, y_te) 
                
                history['train_loss'].append(train_loss) 
                history['train_acc'].append(train_acc) 
                history['test_loss'].append(test_loss) 
                history['test_acc'].append(test_acc) 
                
                if it % 500 == 0:
                    print(f"  Iter {it}: Test Acc={test_acc:.4f}, Test Loss={test_loss:.4f}")
        
        tiempo = time.time() - inicio
        
        # Guardar resultados
        resultados[tam] = {
            'dimensiones': X_tr.shape[1],
            'tiempo': tiempo,
            'train_acc_final': history['train_acc'][-1],
            'test_acc_final': history['test_acc'][-1],
            'train_loss_final': history['train_loss'][-1],
            'test_loss_final': history['test_loss'][-1],
            'history': history
        }
        
        print(f"Tiempo de entrenamiento: {tiempo:.2f} segundos")
        print(f"Accuracy final - Train: {history['train_acc'][-1]:.4f}, Test: {history['test_acc'][-1]:.4f}")
        print(f"Loss final - Train: {history['train_loss'][-1]:.4f}, Test: {history['test_loss'][-1]:.4f}")
    
    return resultados

# ---------- función para graficar análisis de tamaños de imagen ----------
def graficar_analisis_tamanos(resultados_tamanos):
    """
    Grafica los resultados del análisis de diferentes tamaños de imagen
    """
    tamanos = list(resultados_tamanos.keys())
    
    # Extraer datos para graficar
    dimensiones = [resultados_tamanos[tam]['dimensiones'] for tam in tamanos]
    tiempos = [resultados_tamanos[tam]['tiempo'] for tam in tamanos]
    train_acc = [resultados_tamanos[tam]['train_acc_final'] for tam in tamanos]
    test_acc = [resultados_tamanos[tam]['test_acc_final'] for tam in tamanos]
    train_loss = [resultados_tamanos[tam]['train_loss_final'] for tam in tamanos]
    test_loss = [resultados_tamanos[tam]['test_loss_final'] for tam in tamanos]
    
    # Gráfico 1: Accuracy vs Tamaño de imagen
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(tamanos, train_acc, color ='#48cae4', linewidth=2, markersize=8, label='Train')
    plt.plot(tamanos, test_acc, color ='#0077b6', linewidth=2, markersize=8, label='Test')
    plt.title('Accuracy vs Tamaño de Imagen')
    plt.xlabel('Tamaño de imagen (píxeles)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Gráfico 2: Loss vs Tamaño de imagen
    plt.subplot(2, 3, 2)
    plt.plot(tamanos, train_loss, color ='#48cae4', linewidth=2, markersize=8, label='Train')
    plt.plot(tamanos, test_loss, color ='#0077b6', linewidth=2, markersize=8, label='Test')
    plt.title('Loss vs Tamaño de Imagen')
    plt.xlabel('Tamaño de imagen (píxeles)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Gráfico 3: Tiempo de cómputo vs Tamaño de imagen
    plt.subplot(2, 3, 3)
    plt.plot(tamanos, tiempos, color = 'blue', linewidth=2, markersize=8)
    plt.title('Tiempo de Cómputo vs Tamaño de Imagen')
    plt.xlabel('Tamaño de imagen (píxeles)')
    plt.ylabel('Tiempo (segundos)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Tabla resumen
    print("\n" + "="*80)
    print("RESUMEN DEL ANÁLISIS DE TAMAÑOS DE IMAGEN")
    print("="*80)
    print(f"{'Tamaño':<10} {'Dimensiones':<12} {'Tiempo(s)':<10} {'Train Acc':<12} {'Test Acc':<12} {'Train Loss':<12} {'Test Loss':<12}")
    print("-" * 80)
    for tam in tamanos:
        res = resultados_tamanos[tam]
        print(f"{tam:<10} {res['dimensiones']:<12} {res['tiempo']:<10.2f} {res['train_acc_final']:<12.4f} {res['test_acc_final']:<12.4f} {res['train_loss_final']:<12.4f} {res['test_loss_final']:<12.4f}")
    print("="*80)
  
# ---------- descenso por gradiente con historial ----------
def gradient_descent(X_train, D_train, X_test, D_test, alpha, iterations=2000): 

    # Fijar semilla para reproducibilidad
    np.random.seed(42)
    
    K = X_train.shape[1] 
    w = np.random.randn(K) * 0.01 
    b = 0.0
    """
    K = X_train.shape[1]
    w = np.random.randn(K) / np.sqrt(K)       # σ ≈ 1/√K ≈ 0.016
    b = 0.0"""

    history = { 
        'train_loss': [], 
        'train_acc': [], 
        'test_loss': [], 
        'test_acc': [] 
    } 

    for it in range(iterations): 
        grad_w, grad_b = grad_L(w, b, X_train, D_train) 
        w -= alpha * grad_w 
        b -= alpha * grad_b 

        if it % 10 == 0 or it == iterations - 1: 
            train_loss = loss(w, b, X_train, D_train) 
            train_acc = accuracy(w, b, X_train, D_train) 
            test_loss = loss(w, b, X_test, D_test) 
            test_acc = accuracy(w, b, X_test, D_test) 

            history['train_loss'].append(train_loss) 
            history['train_acc'].append(train_acc) 
            history['test_loss'].append(test_loss) 
            history['test_acc'].append(test_acc) 

            print(f"Iter {it}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}") 
        if it % 200 == 0 and it>0:
            alpha *= 0.5          # divide el paso a la mitad
    return w, b, history['train_loss'], history['train_acc'], history['test_loss'], history['test_acc']

def graficar_convergencia_mejor_alpha(loss_tr_hist, loss_te_hist, acc_tr_hist, acc_te_hist, alpha):
    """
    Grafica la convergencia del modelo con el mejor alpha encontrado
    """
    plt.figure(figsize=(15, 5))
    
    # Gráfico 1: Pérdida de entrenamiento
    plt.subplot(1, 3, 1)
    plt.plot(loss_tr_hist, 'b-', linewidth=2, label='Train')
    plt.plot(loss_te_hist, 'r-', linewidth=2, label='Test')
    plt.title(f'Convergencia de Pérdida\n(α = {alpha})')
    plt.xlabel('Iteraciones')
    plt.ylabel('Pérdida')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Gráfico 2: Accuracy de entrenamiento
    plt.subplot(1, 3, 2)
    plt.plot(acc_tr_hist, 'b-', linewidth=2, label='Train')
    plt.title(f'Accuracy de Entrenamiento\n(α = {alpha})')
    plt.xlabel('Iteraciones')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Gráfico 3: Accuracy de test
    plt.subplot(1, 3, 3)
    plt.plot(acc_te_hist, 'b-', linewidth=2, label='Test')
    plt.title(f'Accuracy de Test\n(α = {alpha})')
    plt.xlabel('Iteraciones')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir estadísticas finales
    print(f"\n--- Resultados con α = {alpha} ---")
    print(f"Pérdida Train final: {loss_tr_hist[-1]:.6f}")
    print(f"Pérdida Test final: {loss_te_hist[-1]:.6f}")
    print(f"Accuracy Train final: {acc_tr_hist[-1]:.4f}")
    print(f"Accuracy Test final: {acc_te_hist[-1]:.4f}")
    print(f"Mejora en accuracy: {acc_te_hist[-1] - acc_te_hist[0]:.4f}")

def normalizar_min_max(X_train, X_test):
    """
    Normaliza los datos entre 0 y 1 usando min-max scaling
    """
    min_vals = X_train.min(axis=0)
    max_vals = X_train.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # evitar división por 0
    
    X_train_norm = (X_train - min_vals) / range_vals
    X_test_norm = (X_test - min_vals) / range_vals
    
    return X_train_norm, X_test_norm


# -------------------------------------------ASCENSO POR GRADIENTE-------------------------------------------
def f_sigmoide(x, w, b):
    z = np.dot(x, w) + b
    return 1 / (1 + np.exp(-z))

def loss_sigmoide(w, b, X, D):
    """
    Función de pérdida para el modelo sigmoideo (log-likelihood positivo) - VECTORIZADA
    """
    m = X.shape[0]
    
    # Calcular predicciones vectorizadas
    z = np.dot(X, w) + b
    y_pred = 1 / (1 + np.exp(-z))
    
    # Log-likelihood positivo para maximizar
    # Evitar log(0) y log(1) con epsilon
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Calcular log-likelihood positivo (para maximizar)
    log_likelihood = np.sum(D * np.log(y_pred) + (1 - D) * np.log(1 - y_pred))
    
    return log_likelihood / m

def accuracy_sigmoide(w, b, X, D):
    """
    Calcula la precisión del modelo sigmoideo - VECTORIZADA
    """
    m = X.shape[0]
    
    # Calcular predicciones vectorizadas
    z = np.dot(X, w) + b
    y_pred = 1 / (1 + np.exp(-z))
    
    # Convertir a etiquetas binarias
    y_pred_bin = (y_pred >= 0.5).astype(int)
    
    # Calcular accuracy
    correct = np.sum(y_pred_bin == D)
    
    return correct / m

def grad_sigmoide(w, b, X, D):

    m = X.shape[0]
    
    # Calcular predicciones vectorizadas
    z = np.dot(X, w) + b
    y_pred = 1 / (1 + np.exp(-z))
    
    # Calcular error
    error = D - y_pred
    
    # Calcular gradientes vectorizados
    grad_w = np.dot(X.T, error) / m
    grad_b = np.sum(error) / m
    
    return grad_w, grad_b

def gradient_ascent_sigmoid(X_train, D_train, X_test, D_test, alpha, iterations=2000):
    # Fijar semilla para reproducibilidad
    np.random.seed(42)
    
    K = X_train.shape[1] 
    # Mejor inicialización para ascenso por gradiente
    w = np.random.randn(K) * 0.001  # Pesos más pequeños
    b = 0.0

    history = { 
        'train_loss': [], 
        'train_acc': [], 
        'test_loss': [], 
        'test_acc': [] 
    } 

    for it in range(iterations): 
        grad_w, grad_b = grad_sigmoide(w, b, X_train, D_train) 
        # ASCENSO: sumamos el gradiente para maximizar el log-likelihood positivo
        w += alpha * grad_w 
        b += alpha * grad_b 

        if it % 10 == 0 or it == iterations - 1: 
            train_loss = loss_sigmoide(w, b, X_train, D_train) 
            train_acc = accuracy_sigmoide(w, b, X_train, D_train) 
            test_loss = loss_sigmoide(w, b, X_test, D_test) 
            test_acc = accuracy_sigmoide(w, b, X_test, D_test) 

            history['train_loss'].append(train_loss) 
            history['train_acc'].append(train_acc) 
            history['test_loss'].append(test_loss) 
            history['test_acc'].append(test_acc) 

            print(f"Iter {it}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}") 
        
    
    return w, b, history['train_loss'], history['train_acc'], history['test_loss'], history['test_acc']

def predecir_sigmoid(X, w, b):
    m = X.shape[0]
    y_pred_prob = np.zeros(m)
    y_pred_bin = np.zeros(m)
    
    for i in range(m):
        y_pred_prob[i] = f_sigmoide(X[i], w, b)
        y_pred_bin[i] = 1 if y_pred_prob[i] >= 0.5 else 0
    
    return y_pred_bin, y_pred_prob

def graficar_convergencia_sigmoid(loss_tr_hist, loss_te_hist, acc_tr_hist, acc_te_hist, alpha):
    """
    Grafica la convergencia del modelo con ascenso por gradiente sigmoideo
    """
    plt.figure(figsize=(15, 6))
    
    # Gráfico 1: Accuracy de entrenamiento
    plt.subplot(1, 2, 1)
    plt.plot(acc_tr_hist, color = '#132a13', linewidth=2, label='Train')
    plt.title(f'Accuracy de Entrenamiento (Ascenso Sigmoid)\n(α = {alpha})')
    plt.xlabel('Iteraciones')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Gráfico 2: Accuracy de testing
    plt.subplot(1, 2, 2)
    plt.plot(acc_te_hist, color = '#4f772d', linewidth=2, label='Test')
    plt.title(f'Accuracy de Testing (Ascenso Sigmoid)\n(α = {alpha})')
    plt.xlabel('Iteraciones')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir estadísticas finales
    print(f"\n--- Resultados Ascenso por Gradiente Sigmoid (α = {alpha}) ---")
    print(f"Accuracy Train final: {acc_tr_hist[-1]:.4f}")
    print(f"Accuracy Test final: {acc_te_hist[-1]:.4f}")
    print(f"Mejora en accuracy de testing: {acc_te_hist[-1] - acc_te_hist[0]:.4f}")

def comparar_sigmoid_alpha(X_train, D_train, X_test, D_test, alpha1=0.001, alpha2=0.01, iterations=2000):
    """
    Compara los resultados del ascenso por gradiente sigmoideo con dos valores de alpha diferentes
    """
    print(f"Comparando ascenso por gradiente sigmoideo con α = {alpha1} vs α = {alpha2}")
    print("="*60)
    
    # Fijar semilla para reproducibilidad
    np.random.seed(42)
    
    # Entrenar con primer alpha
    print(f"\nEntrenando con α = {alpha1}...")
    w1, b1, loss_tr1, acc_tr1, loss_te1, acc_te1 = gradient_ascent_sigmoid(X_train, D_train, X_test, D_test, alpha1, iterations)
    
    # Fijar semilla nuevamente para reproducibilidad
    np.random.seed(42)
    
    # Entrenar con segundo alpha
    print(f"\nEntrenando con α = {alpha2}...")
    w2, b2, loss_tr2, acc_tr2, loss_te2, acc_te2 = gradient_ascent_sigmoid(X_train, D_train, X_test, D_test, alpha2, iterations)
    
    # Graficar comparación
    plt.figure(figsize=(15, 6))
    
    # Gráfico 1: Accuracy de entrenamiento
    plt.subplot(1, 2, 1)
    plt.plot(acc_tr1, color = '#132a13', linewidth=2, label=f'α = {alpha1}')
    plt.plot(acc_tr2, color = '#4f772d', linewidth=2, label=f'α = {alpha2}')
    plt.title('Accuracy de Entrenamiento - Comparación Ascenso Sigmoid')
    plt.xlabel('Iteraciones')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Gráfico 2: Accuracy de testing
    plt.subplot(1, 2, 2)
    plt.plot(acc_te1, color = '#132a13', linewidth=2, label=f'α = {alpha1}')
    plt.plot(acc_te2, color = '#4f772d', linewidth=2, label=f'α = {alpha2}')
    plt.title('Accuracy de Testing - Comparación Ascenso Sigmoid')
    plt.xlabel('Iteraciones')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir estadísticas comparativas
    print(f"\n--- Comparación Ascenso por Gradiente Sigmoid ---")
    print(f"α = {alpha1}:")
    print(f"  Accuracy Train final: {acc_tr1[-1]:.4f}")
    print(f"  Accuracy Test final: {acc_te1[-1]:.4f}")
    print(f"  Mejora en accuracy: {acc_te1[-1] - acc_te1[0]:.4f}")
    
    print(f"\nα = {alpha2}:")
    print(f"  Accuracy Train final: {acc_tr2[-1]:.4f}")
    print(f"  Accuracy Test final: {acc_te2[-1]:.4f}")
    print(f"  Mejora en accuracy: {acc_te2[-1] - acc_te2[0]:.4f}")
    
    # Determinar cuál es mejor
    if acc_te1[-1] > acc_te2[-1]:
        mejor_alpha = alpha1
        mejor_acc = acc_te1[-1]
        print(f"\nMEJOR RESULTADO: α = {mejor_alpha} (Accuracy Test: {mejor_acc:.4f})")
    elif acc_te2[-1] > acc_te1[-1]:
        mejor_alpha = alpha2
        mejor_acc = acc_te2[-1]
        print(f"\nMEJOR RESULTADO: α = {mejor_alpha} (Accuracy Test: {mejor_acc:.4f})")
    else:
        print(f"\nRESULTADOS SIMILARES: Ambos α obtienen accuracy similar")

def comparar_descenso_ascenso(X_train, D_train, X_test, D_test, alpha1=0.001, alpha2=0.01, iterations=2000, 
                             desc1_results=None, asc1_results=None, desc2_results=None, asc2_results=None):
    print(f"Comparando descenso vs ascenso por gradiente con α = {alpha1} y α = {alpha2}")
    print("="*70)
    
    # Reutilizar resultados si están disponibles, sino calcular
    if desc1_results is None:
        print(f"\nEntrenando DESCENSO con α = {alpha1}...")
        w_desc1, b_desc1, loss_tr_desc1, acc_tr_desc1, loss_te_desc1, acc_te_desc1 = gradient_descent(X_train, D_train, X_test, D_test, alpha1, iterations)
    else:
        print(f"\nReutilizando resultados DESCENSO con α = {alpha1}...")
        w_desc1, b_desc1, loss_tr_desc1, acc_tr_desc1, loss_te_desc1, acc_te_desc1 = desc1_results
    
    if asc1_results is None:
        print(f"\nEntrenando ASCENSO con α = {alpha1}...")
        w_asc1, b_asc1, loss_tr_asc1, acc_tr_asc1, loss_te_asc1, acc_te_asc1 = gradient_ascent_sigmoid(X_train, D_train, X_test, D_test, alpha1, iterations)
    else:
        print(f"\nReutilizando resultados ASCENSO con α = {alpha1}...")
        w_asc1, b_asc1, loss_tr_asc1, acc_tr_asc1, loss_te_asc1, acc_te_asc1 = asc1_results
    
    if desc2_results is None:
        print(f"\nEntrenando DESCENSO con α = {alpha2}...")
        w_desc2, b_desc2, loss_tr_desc2, acc_tr_desc2, loss_te_desc2, acc_te_desc2 = gradient_descent(X_train, D_train, X_test, D_test, alpha2, iterations)
    else:
        print(f"\nReutilizando resultados DESCENSO con α = {alpha2}...")
        w_desc2, b_desc2, loss_tr_desc2, acc_tr_desc2, loss_te_desc2, acc_te_desc2 = desc2_results
    
    if asc2_results is None:
        print(f"\nEntrenando ASCENSO con α = {alpha2}...")
        w_asc2, b_asc2, loss_tr_asc2, acc_tr_asc2, loss_te_asc2, acc_te_asc2 = gradient_ascent_sigmoid(X_train, D_train, X_test, D_test, alpha2, iterations)
    else:
        print(f"\nReutilizando resultados ASCENSO con α = {alpha2}...")
        w_asc2, b_asc2, loss_tr_asc2, acc_tr_asc2, loss_te_asc2, acc_te_asc2 = asc2_results
    
    # Graficar comparación
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Gráfico 1: Pérdida de entrenamiento
    axes[0, 0].plot(loss_tr_desc1, color='blue', linewidth=2, label=f'Descenso α={alpha1}')
    axes[0, 0].plot(loss_tr_desc2, color='#0077b6', linewidth=2, label=f'Descenso α={alpha2}')
    axes[0, 0].plot(loss_tr_asc1, color='#132a13', linewidth=2, label=f'Ascenso α={alpha1}')
    axes[0, 0].plot(loss_tr_asc2, color='#4f772d', linewidth=2, label=f'Ascenso α={alpha2}')
    axes[0, 0].set_title('Pérdida de Entrenamiento', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Iteraciones')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Gráfico 2: Pérdida de testing
    axes[0, 1].plot(loss_te_desc1, color='blue', linewidth=2, label=f'Descenso α={alpha1}')
    axes[0, 1].plot(loss_te_desc2, color='#0077b6', linewidth=2, label=f'Descenso α={alpha2}')
    axes[0, 1].plot(loss_te_asc1, color='#132a13', linewidth=2, label=f'Ascenso α={alpha1}')
    axes[0, 1].plot(loss_te_asc2, color='#4f772d', linewidth=2, label=f'Ascenso α={alpha2}')
    axes[0, 1].set_title('Pérdida de Testing', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Iteraciones')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Gráfico 3: Accuracy de entrenamiento
    axes[1, 0].plot(acc_tr_desc1, color='blue', linewidth=2, label=f'Descenso α={alpha1}')
    axes[1, 0].plot(acc_tr_desc2, color='#0077b6', linewidth=2, label=f'Descenso α={alpha2}')
    axes[1, 0].plot(acc_tr_asc1, color='#132a13', linewidth=2, label=f'Ascenso α={alpha1}')
    axes[1, 0].plot(acc_tr_asc2, color='#4f772d', linewidth=2, label=f'Ascenso α={alpha2}')
    axes[1, 0].set_title('Accuracy de Entrenamiento', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Iteraciones')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Gráfico 4: Accuracy de testing
    axes[1, 1].plot(acc_te_desc1, color='blue', linewidth=2, label=f'Descenso α={alpha1}')
    axes[1, 1].plot(acc_te_desc2, color='#0077b6', linewidth=2, label=f'Descenso α={alpha2}')
    axes[1, 1].plot(acc_te_asc1, color='#132a13', linewidth=2, label=f'Ascenso α={alpha1}')
    axes[1, 1].plot(acc_te_asc2, color='#4f772d', linewidth=2, label=f'Ascenso α={alpha2}')
    axes[1, 1].set_title('Accuracy de Testing', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Iteraciones')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout(pad=3.0)
    plt.show()
    
    # Imprimir estadísticas comparativas
    print(f"\n--- Comparación Descenso vs Ascenso por Gradiente ---")
    print(f"DESCENSO α = {alpha1}:")
    print(f"  Loss Train final: {loss_tr_desc1[-1]:.4f}")
    print(f"  Loss Test final: {loss_te_desc1[-1]:.4f}")
    print(f"  Accuracy Train final: {acc_tr_desc1[-1]:.4f}")
    print(f"  Accuracy Test final: {acc_te_desc1[-1]:.4f}")
    
    print(f"\nDESCENSO α = {alpha2}:")
    print(f"  Loss Train final: {loss_tr_desc2[-1]:.4f}")
    print(f"  Loss Test final: {loss_te_desc2[-1]:.4f}")
    print(f"  Accuracy Train final: {acc_tr_desc2[-1]:.4f}")
    print(f"  Accuracy Test final: {acc_te_desc2[-1]:.4f}")
    
    print(f"\nASCENSO α = {alpha1}:")
    print(f"  Loss Train final: {loss_tr_asc1[-1]:.4f}")
    print(f"  Loss Test final: {loss_te_asc1[-1]:.4f}")
    print(f"  Accuracy Train final: {acc_tr_asc1[-1]:.4f}")
    print(f"  Accuracy Test final: {acc_te_asc1[-1]:.4f}")
    
    print(f"\nASCENSO α = {alpha2}:")
    print(f"  Loss Train final: {loss_tr_asc2[-1]:.4f}")
    print(f"  Loss Test final: {loss_te_asc2[-1]:.4f}")
    print(f"  Accuracy Train final: {acc_tr_asc2[-1]:.4f}")
    print(f"  Accuracy Test final: {acc_te_asc2[-1]:.4f}")
    
    # Determinar el mejor método
    mejores_resultados = {
        'descenso_alpha1': acc_te_desc1[-1],
        'descenso_alpha2': acc_te_desc2[-1],
        'ascenso_alpha1': acc_te_asc1[-1],
        'ascenso_alpha2': acc_te_asc2[-1]
    }
    
    mejor_metodo = max(mejores_resultados, key=mejores_resultados.get)
    mejor_acc = mejores_resultados[mejor_metodo]
    
    print(f"\nMEJOR RESULTADO: {mejor_metodo} (Accuracy Test: {mejor_acc:.4f})")

# ---------- main ----------
def main():
    # Fijar semilla para reproducibilidad
    np.random.seed(42)
    
    path_h = "../Healthy"
    path_p = "../Parkinson"


    X, y = cargar_datos(path_h, path_p, max_imgs=300, tam=(64, 64)) 
    X_tr, X_te, y_tr, y_te = dividir_datos(X, y)

    print("Usando normalización Min-Max (0-1)")
    X_tr_norm, X_te_norm = normalizar_min_max(X_tr, X_te)

    # ---------------- DESCENSO POR GRADIENTE ----------------
    """
    print("Entrenando sin normalizar ...")
    w, b, loss_hist, acc_tr_hist, loss_te_hist, acc_te_hist = gradient_descent(X_tr, y_tr, X_te, y_te, 0.002)

    print("Entrenando con datos normalizados ...")
    w, b, loss_hist_norm, acc_tr_hist_norm, loss_te_hist_norm, acc_te_hist_norm = gradient_descent(X_tr_norm, y_tr, X_te_norm, y_te, 0.002)
  
    #---------------- Gráficos ----------------    
    # Imagen 1: Pérdida de entrenamiento
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(loss_hist_norm, 'b-', label="Normalizado")
    ax1.set_title("Pérdida - Entrenamiento (Normalizado)")
    ax1.set_xlabel("Iteraciones (cada 10)")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(loss_hist,color ='#0077b6', label="Sin normalizar")
    ax2.set_title("Pérdida - Entrenamiento (Sin normalizar)")
    ax2.set_xlabel("Iteraciones (cada 10)")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Imagen 2: Pérdida de testing
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax3.plot(loss_te_hist_norm, 'b-', label="Normalizado")
    ax3.set_title("Pérdida - Testing (Normalizado)")
    ax3.set_xlabel("Iteraciones (cada 10)")
    ax3.set_ylabel("Loss")
    ax3.legend()
    ax3.grid(True)
    
    ax4.plot(loss_te_hist, color ='#0077b6', label="Sin normalizar")
    ax4.set_title("Pérdida - Testing (Sin normalizar)")
    ax4.set_xlabel("Iteraciones (cada 10)")
    ax4.set_ylabel("Loss")
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Imagen 3: Exactitud de test
    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax5.plot(acc_te_hist_norm, 'b-', label="Normalizado")
    ax5.set_title("Exactitud - Test (Normalizado)")
    ax5.set_xlabel("Iteraciones (cada 10)")
    ax5.set_ylabel("Accuracy")
    ax5.set_ylim(0, 1)
    ax5.legend()
    ax5.grid(True)
    
    ax6.plot(acc_te_hist, color ='#0077b6', label="Sin normalizar")
    ax6.set_title("Exactitud - Test (Sin normalizar)")
    ax6.set_xlabel("Iteraciones (cada 10)")
    ax6.set_ylabel("Accuracy")
    ax6.set_ylim(0, 1)
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.show() 

    # Imagen 4: Exactitud de entrenamiento
    fig4, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax5.plot(acc_tr_hist_norm, 'b-', label="Normalizado")
    ax5.set_title("Exactitud - Entrenamiento (Normalizado)")
    ax5.set_xlabel("Iteraciones (cada 10)")
    ax5.set_ylabel("Accuracy")
    ax5.set_ylim(0, 1)
    ax5.legend()
    ax5.grid(True)
    
    ax6.plot(acc_tr_hist, color ='#0077b6', label="Sin normalizar")
    ax6.set_title("Exactitud - Entrenamiento (Sin normalizar)")
    ax6.set_xlabel("Iteraciones (cada 10)")
    ax6.set_ylabel("Accuracy")
    ax6.set_ylim(0, 1)
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.show() 
    """
    """
    # ---------- Análisis del impacto del parámetro α en la convergencia ----------
    print("\n" + "="*50)
    print("ANÁLISIS DEL IMPACTO DEL PARÁMETRO α EN LA CONVERGENCIA")
    print("="*50)
    
    # Analizar convergencia para diferentes valores de α
    resultados_convergencia = analizar_convergencia_alpha(X_tr_norm, y_tr, X_te_norm, y_te, 
                                                        alphas=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    
        # Graficar resultados de convergencia
    graficar_convergencia_alpha(resultados_convergencia)


    # ---------- Análisis del impacto del tamaño de imagen ----------
    print("\n" + "="*50)
    print("ANÁLISIS DEL IMPACTO DEL TAMAÑO DE IMAGEN")
    print("="*50)
    
    # Análisis con diferentes tamaños
    resultados_tamanos = analizar_tamanos_imagen(path_h, path_p, tamanos=[16, 32, 64, 128])
    
    # ---------- Gráficos del análisis de tamaños ----------
    graficar_analisis_tamanos(resultados_tamanos)

    # ---------- Predicción y matriz de confusión ----------
    print("\n" + "="*50)
    print("PREDICCIÓN Y ANÁLISIS DE EFECTIVIDAD")
    print("="*50)
    
    # Predecir etiquetas para el conjunto de testing
    y_pred_bin, y_pred_prob = predecir_etiquetas(X_te_norm, w, b)
    
    # Generar matriz de confusión y métricas
    cm, accuracy, report = generar_matriz_confusion(y_te, y_pred_bin)
    
    # Mostrar resultados
    print(f"\nAccuracy en testing: {accuracy:.4f}")
    print(f"\nMatriz de Confusión:")
    print(cm)
    print(f"\nReporte de Clasificación:")
    print(report)
    
    # Graficar matriz de confusión
    graficar_matriz_confusion(cm, accuracy, "Matriz de Confusión - Datos Normalizados")
    
    """
    # ---------- ASCENSO POR GRADIENTE SIGMOIDEO ----------
    print("\n" + "="*50)
    print("ASCENSO POR GRADIENTE CON FUNCIÓN SIGMOIDEA")
    print("="*50)
    #Utilizamos los datos ya cargados y normalizados de 64 x 64
    w_sigmoid, b_sigmoid, loss_tr_sigmoid, acc_tr_sigmoid, loss_te_sigmoid, acc_te_sigmoid = gradient_ascent_sigmoid(X_tr_norm, y_tr, X_te_norm, y_te, 0.001)
    
    # Graficar convergencia del ascenso sigmoideo
    #print("\n--- Gráficos de convergencia del ascenso por gradiente sigmoideo ---")
    #graficar_convergencia_sigmoid(loss_tr_sigmoid, loss_te_sigmoid, acc_tr_sigmoid, acc_te_sigmoid, 0.001)

    #-------------- Convergencia de α --------------
    #Los dos mejores valores de α para el descenso por gradiente son 0.001 y 0.01
    #comparar_sigmoid_alpha(X_tr_norm, y_tr, X_te_norm, y_te, 0.001, 0.01)

    #-------------- Comparar  --------------
    
    comparar_descenso_ascenso(X_tr_norm, y_tr, X_te_norm, y_te, 0.001, 0.01)

if __name__ == "__main__":
    main()

