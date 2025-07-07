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
    
    # Crear reporte manual
    report = f"""
Reporte de Clasificación:
              precision    recall  f1-score   support

Healthy          {tn/(tn+fn):.3f}      {tn/(tn+fp):.3f}      {2*(tn/(tn+fn))*(tn/(tn+fp))/((tn/(tn+fn))+(tn/(tn+fp))):.3f}        {tn+fp}
Parkinson        {precision:.3f}      {recall:.3f}      {f1:.3f}       {tp+fn}

accuracy                            {accuracy:.3f}      {len(y_true)}
macro avg        {(tn/(tn+fn) + precision)/2:.3f}      {(tn/(tn+fp) + recall)/2:.3f}      {((2*(tn/(tn+fn))*(tn/(tn+fp))/((tn/(tn+fn))+(tn/(tn+fp)))) + f1)/2:.3f}       {len(y_true)}
weighted avg     {(tn/(tn+fn) + precision)/2:.3f}      {(tn/(tn+fp) + recall)/2:.3f}      {((2*(tn/(tn+fn))*(tn/(tn+fp))/((tn/(tn+fn))+(tn/(tn+fp)))) + f1)/2:.3f}       {len(y_true)}
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

# ---------- análisis de diferentes learning rates ----------
def analizar_learning_rates(X_train, D_train, X_test, D_test, alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1e-0], iterations=1000):
    """
    Analiza el impacto de diferentes valores de α en la convergencia
    """
    resultados = {}
    
    for alpha in alphas:
        print(f"\nEntrenando con α = {alpha}")
        
        # Entrenar modelo con este alpha
        K = X_train.shape[1] 
        w = np.random.randn(K) * 0.01 
        b = 0.0 
        
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
            
            if it % 100 == 0 or it == iterations - 1: 
                train_loss = loss(w, b, X_train, D_train) 
                train_acc = accuracy(w, b, X_train, D_train) 
                test_loss = loss(w, b, X_test, D_test) 
                test_acc = accuracy(w, b, X_test, D_test) 
                
                history['train_loss'].append(train_loss) 
                history['train_acc'].append(train_acc) 
                history['test_loss'].append(test_loss) 
                history['test_acc'].append(test_acc) 
                
                if it % 500 == 0:
                    print(f"  Iter {it}: Test Acc={test_acc:.4f}, Test Loss={test_loss:.4f}")
        
        resultados[alpha] = history
        print(f"  Accuracy final: {history['test_acc'][-1]:.4f}")
    
    return resultados

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

# ---------- función para graficar análisis de α (solo normalizado) ----------
def graficar_analisis_alpha_normalizado(resultados_norm):
    """
    Grafica los resultados del análisis de diferentes learning rates (solo datos normalizados)
    """
    alphas = list(resultados_norm.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Gráfico 1: Accuracy final vs α
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    acc_finales = [resultados_norm[alpha]['test_acc'][-1] for alpha in alphas]
    plt.semilogx(alphas, acc_finales, 'bo-', linewidth=2, markersize=8)
    plt.title('Accuracy Final vs α (Datos Normalizados)')
    plt.xlabel('Learning Rate (α)')
    plt.ylabel('Accuracy Final')
    plt.grid(True)
    
    # Gráfico 2: Loss final vs α
    plt.subplot(2, 3, 2)
    loss_finales = [resultados_norm[alpha]['test_loss'][-1] for alpha in alphas]
    plt.semilogx(alphas, loss_finales, 'ro-', linewidth=2, markersize=8)
    plt.title('Loss Final vs α (Datos Normalizados)')
    plt.xlabel('Learning Rate (α)')
    plt.ylabel('Loss Final')
    plt.grid(True)
    
    # Gráfico 3: Convergencia de Accuracy
    plt.subplot(2, 3, 3)
    for i, alpha in enumerate(alphas):
        plt.plot(resultados_norm[alpha]['test_acc'], 
                color=colors[i], label=f'α={alpha}')
    plt.title('Convergencia de Accuracy vs α')
    plt.xlabel('Iteraciones (cada 100)')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Gráfico 4: Convergencia de Loss
    plt.subplot(2, 3, 4)
    for i, alpha in enumerate(alphas):
        plt.plot(resultados_norm[alpha]['test_loss'], 
                color=colors[i], label=f'α={alpha}')
    plt.title('Convergencia de Loss vs α')
    plt.xlabel('Iteraciones (cada 100)')
    plt.ylabel('Test Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Tabla resumen
    print("\n" + "="*60)
    print("RESUMEN DEL ANÁLISIS DE α (DATOS NORMALIZADOS)")
    print("="*60)
    print(f"{'α':<10} {'Train Acc':<12} {'Test Acc':<12} {'Train Loss':<12} {'Test Loss':<12}")
    print("-" * 60)
    for alpha in alphas:
        train_acc = resultados_norm[alpha]['train_acc'][-1]
        test_acc = resultados_norm[alpha]['test_acc'][-1]
        train_loss = resultados_norm[alpha]['train_loss'][-1]
        test_loss = resultados_norm[alpha]['test_loss'][-1]
        print(f"{alpha:<10} {train_acc:<12.4f} {test_acc:<12.4f} {train_loss:<12.4f} {test_loss:<12.4f}")
    print("="*60)

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
    plt.plot(tamanos, train_acc, 'bo-', linewidth=2, markersize=8, label='Train')
    plt.plot(tamanos, test_acc, 'ro-', linewidth=2, markersize=8, label='Test')
    plt.title('Accuracy vs Tamaño de Imagen')
    plt.xlabel('Tamaño de imagen (píxeles)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Gráfico 2: Loss vs Tamaño de imagen
    plt.subplot(2, 3, 2)
    plt.plot(tamanos, train_loss, 'bo-', linewidth=2, markersize=8, label='Train')
    plt.plot(tamanos, test_loss, 'ro-', linewidth=2, markersize=8, label='Test')
    plt.title('Loss vs Tamaño de Imagen')
    plt.xlabel('Tamaño de imagen (píxeles)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Gráfico 3: Tiempo de cómputo vs Tamaño de imagen
    plt.subplot(2, 3, 3)
    plt.plot(tamanos, tiempos, 'go-', linewidth=2, markersize=8)
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
    return w, b, history['train_loss'], history['train_acc'], history['test_acc']

def graficar_convergencia_mejor_alpha(loss_hist, acc_tr_hist, acc_te_hist, alpha):
    """
    Grafica la convergencia del modelo con el mejor alpha encontrado
    """
    plt.figure(figsize=(15, 5))
    
    # Gráfico 1: Pérdida
    plt.subplot(1, 3, 1)
    plt.plot(loss_hist, 'b-', linewidth=2, label='Pérdida')
    plt.title(f'Convergencia de Pérdida\n(α = {alpha})')
    plt.xlabel('Iteraciones')
    plt.ylabel('Pérdida')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Gráfico 2: Accuracy de entrenamiento
    plt.subplot(1, 3, 2)
    plt.plot(acc_tr_hist, 'g-', linewidth=2, label='Train')
    plt.title(f'Accuracy de Entrenamiento\n(α = {alpha})')
    plt.xlabel('Iteraciones')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Gráfico 3: Accuracy de test
    plt.subplot(1, 3, 3)
    plt.plot(acc_te_hist, 'r-', linewidth=2, label='Test')
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
    print(f"Pérdida final: {loss_hist[-1]:.6f}")
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



# ---------- main ----------
def main():
    path_h = "../Healthy"
    path_p = "../Parkinson"
    """path_h = "C:\Users\casa\Downloads\Metodos\Healthy"
    path_p = "C:\Users\casa\Downloads\Metodos\Parkinson"""

    X, y = cargar_datos(path_h, path_p, max_imgs=300, tam=(64, 64))
    X_tr, X_te, y_tr, y_te = dividir_datos(X, y)

    print("Entrenando ...")
    w, b, loss_hist, acc_tr_hist, acc_te_hist = gradient_descent(X_tr, y_tr, X_te, y_te, 0.0025)

    print("Entrenando con datos normalizados ...")
    
    print("Usando normalización Min-Max (0-1)")
    X_tr_norm, X_te_norm = normalizar_min_max(X_tr, X_te)

    w, b, loss_hist_norm, acc_tr_hist_norm, acc_te_hist_norm = gradient_descent(X_tr_norm, y_tr, X_te_norm, y_te, 0.0025)
 #---------------- Gráficos ----------------    
    # Imagen 1: Pérdida de entrenamiento
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(loss_hist_norm, 'b-', label="Normalizado")
    ax1.set_title("Pérdida - Entrenamiento (Normalizado)")
    ax1.set_xlabel("Iteraciones (cada 10)")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(loss_hist, 'r-', label="Sin normalizar")
    ax2.set_title("Pérdida - Entrenamiento (Sin normalizar)")
    ax2.set_xlabel("Iteraciones (cada 10)")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Imagen 2: Exactitud de entrenamiento
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax3.plot(acc_tr_hist_norm, 'b-', label="Normalizado")
    ax3.set_title("Exactitud - Entrenamiento (Normalizado)")
    ax3.set_xlabel("Iteraciones (cada 10)")
    ax3.set_ylabel("Accuracy")
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True)
    
    ax4.plot(acc_tr_hist, 'r-', label="Sin normalizar")
    ax4.set_title("Exactitud - Entrenamiento (Sin normalizar)")
    ax4.set_xlabel("Iteraciones (cada 10)")
    ax4.set_ylabel("Accuracy")
    ax4.set_ylim(0, 1)
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
    
    ax6.plot(acc_te_hist, 'r-', label="Sin normalizar")
    ax6.set_title("Exactitud - Test (Sin normalizar)")
    ax6.set_xlabel("Iteraciones (cada 10)")
    ax6.set_ylabel("Accuracy")
    ax6.set_ylim(0, 1)
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.show() 
    """


  # ---------- Análisis del impacto de α ----------
    print("\n" + "="*50)
    print("ANÁLISIS DEL IMPACTO DEL PARÁMETRO α")
    print("="*50)
    
    # Análisis con datos normalizados
    print("\n--- Análisis con datos NORMALIZADOS ---")
    resultados_norm = analizar_learning_rates(X_tr_norm, y_tr, X_te_norm, y_te, 
                                             alphas=[1e-7, 1e-6, 5e-6, 1e-5, 5e-5])


    # ---------- Gráficos del análisis de α ----------
    graficar_analisis_alpha_normalizado(resultados_norm)
   
        #---------------Calculamos la convergencia con el mejor alpha-----------------
    print("\n" + "="*50)
    print("CALCULAMOS LA CONVERGENCIA CON EL MEJOR ALPHA")
    print("="*50)
    w, b, loss_hist_norm, acc_tr_hist_norm, acc_te_hist_norm = gradient_descent(X_tr_norm, y_tr, X_te_norm, y_te, 5e-05)
    graficar_convergencia_mejor_alpha(loss_hist_norm, acc_tr_hist_norm, acc_te_hist_norm, 5e-05)
    
    """

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
    
if __name__ == "__main__":
    main()
