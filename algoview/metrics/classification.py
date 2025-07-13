import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve, auc

def evaluate_model(y_true, y_pred, y_prob=None, model_name='Model'):
    """
    Оценка качества модели классификации.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob) if y_prob is not None else None

    print(f"{model_name} — Accuracy: {acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, AUC-ROC: {auc_roc:.4f}" if auc_roc else
          f"{model_name} — Accuracy: {acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}")

    if y_prob is not None:
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc_roc:.3f})')
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()

        # Precision-Recall curve
        precision, recall_vals, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall_vals, precision)

        plt.subplot(1,2,2)
        plt.plot(recall_vals, precision, label=f'{model_name} (AUC={pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Возвращаем метрики в словаре
    return {
        'Accuracy': acc,
        'F1': f1,
        'Recall': recall,
        'AUC-ROC': auc_roc
    } 