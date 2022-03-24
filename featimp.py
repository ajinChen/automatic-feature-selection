import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# data proprocessing
def load_data(Path):
    data = pd.read_csv(Path, index_col=0)
    col = 'diagnosis'
    data[col] = data[col].astype('category').cat.as_ordered()
    encoder = data[col].cat.categories
    data[col] = data[col].cat.codes
    data = data.drop(columns=['Unnamed: 32'])
    return data


def split_target(df, target):
    Y = df[target].values
    X = df.drop(columns=[target])
    return X, Y


def train_val_split(df, ratio):
    train, val = train_test_split(df, train_size=ratio, shuffle=True)
    return train, val


def label_encoding_with_NAs(train, val, col):
    train[col] = train[col].astype('category').cat.as_ordered()
    encoder = train[col].cat.categories
    train[col] = train[col].cat.codes + 1
    val[col] = pd.Categorical(val[col], categories=encoder, ordered=True)
    val[col] = val[col].cat.codes + 1


    
# Data-based importance strategies
def top_rank(df, target, n=None, ascending=False, method='spearman'):
    """
    Calculate first / last N correlation with target
    This method is measuring single-feature relevance importance and works well for independent features
    But suffers in the presence of codependent features.
    pearson : standard correlation coefficient
    kendall : Kendall Tau correlation coefficient
    spearman : Spearman rank correlation
    :return:
    """
    if not n:
        n = len(df.columns)
    if method == 'PCA':
        scaler = StandardScaler()
        feas = [col for col in df.columns if col != target]
        X = scaler.fit_transform(df.loc[:, feas])
        pca = PCA(n_components=0.9)
        pca.fit(X)
        featimp = {feas[i]:abs(pca.components_[0])[i] for i in range(len(feas))}
        feas = sorted(featimp, key=featimp.get, reverse=True)[:n]
        vals = [featimp[fea] for fea in feas]
    
    else:
        feas = list(abs(df.corr(method=method)[target]).sort_values(ascending=ascending).index[1:n+1])
        vals = list(abs(df.corr(method=method)[target]).sort_values(ascending=ascending))[1:n+1]
    return feas, vals


def mRMR(df, target, mode='spearman', n=None, info=False):
    if not n:
        n = len(df.columns)
    mrmr = dict()
    
    # use different mode to calculate correaltion
    feas, imps = top_rank(df, target, method=mode)
    corr = dict([(fea, imp) for imp, fea in zip(imps, feas)])
    selected_feat = [feas[0]]
    
    for i in range(len(feas)):
        rest_feat = [col for col in feas if col not in selected_feat]
        if not len(rest_feat):
            break
        candi_mrmr = []
        for fi in rest_feat:
            redundancy = 0
            relevance = corr[fi]
            for fj in selected_feat:
                feas, imps = top_rank(df.drop(columns=target), fj, method=mode)
                corr_fj = dict([(fea, imp) for imp, fea in zip(imps, feas)])
                redundancy += corr_fj[fi]
            redundancy /= len(selected_feat)
            candi_mrmr.append(relevance - redundancy)
        max_feature = rest_feat[np.argmax(candi_mrmr)]
        mrmr[max_feature] = np.max(candi_mrmr)
        if info:
            print(f'{i+1} iteration, selected {max_feature} feature with mRMR value = {mrmr[max_feature]:.3f}')
        selected_feat.append(max_feature)
    feat_imp = pd.Series(mrmr.values(), index=mrmr.keys()).sort_values(ascending=False)
    return feat_imp.values[:n], feat_imp.index[:n]


# Model-based importance strategies
def permutation_importance(X_train, y_train, X_valid, y_valid, mode='R'):
    model = rf_model(X_train, y_train, mode)
    if mode == 'R':
        baseline = r2_score(y_valid, model.predict(X_valid))
    else:
        baseline = log_loss(y_valid, model.predict_proba(X_valid))
    imp = []
    for col in X_valid.columns:
        save = X_valid[col].copy()
        X_valid[col] = np.random.permutation(X_valid[col])
        if mode == 'R':
            m = r2_score(y_valid, model.predict(X_valid))
        else:
            m = log_loss(y_valid, model.predict_proba(X_valid))
        X_valid[col] = save
        imp.append(baseline - m)
    feat_imp = pd.Series(imp, index=X_valid.columns).sort_values(ascending=False)
    return feat_imp.values, feat_imp.index


def dropcol_importances(X_train, y_train, X_valid, y_valid, mode='R'):
    model = rf_model(X_train, y_train, mode)
    baseline = model.oob_score_
    imp = []
    for col in X_train.columns:
        X_train_ = X_train.drop(col, axis=1)
        X_valid_ = X_valid.drop(col, axis=1)
        model_ = clone(model)
        model_.fit(X_train_, y_train)
        m = model_.oob_score_
        imp.append(baseline - m)
    feat_imp = pd.Series(imp, index=X_valid.columns).sort_values(ascending=False)
    return feat_imp.values, feat_imp.index


def shap_importances(x_train, y_train, x_val, y_val):
    rf = rf_model(x_train, y_train, mode='R')
    shap_values = shap.TreeExplainer(rf, data=x_train).shap_values(X=x_val, y=y_val, check_additivity=False)
    imp = np.mean(np.abs(shap_values), axis=0)
    return imp, x_val.columns


def rf_model(x_train, y_train, mode='R'):
    hyper = {'min_samples_leaf':80, 'max_features':0.5, 'max_depth':15}
    if mode == 'R':
        rf = RandomForestRegressor(n_estimators=50,
                                 min_samples_leaf=hyper['min_samples_leaf'],
                                 max_features=hyper['max_features'],
                                 max_depth=hyper['max_depth'],
                                 oob_score=True,
                                 n_jobs=-1)
    else:
        rf = RandomForestClassifier(n_estimators=50,
                                 min_samples_leaf=hyper['min_samples_leaf'],
                                 max_features=hyper['max_features'],
                                 max_depth=hyper['max_depth'],
                                 oob_score=True,
                                 n_jobs=-1)
    rf.fit(x_train, y_train)
    return rf



# Visualizing importances
def plot_feature_importances(importances, columns, title, n=None, size=(15, 15), show_values=False, show_var=[0]):
    if not n:
        n = len(columns)
    n_importances = pd.Series(importances, index=columns).sort_values(ascending=True)[-n:]
    fig, ax = plt.subplots(figsize=size)
    if not any(show_var):
        n_importances.plot.barh(color='#4daf4a')
    else:
        ax.barh(n_importances.index, 
            n_importances, 
            xerr=sorted(show_var), color='#4daf4a')
    if show_values:
        for i, bar in enumerate(ax.patches):
            if bar.get_width() < 0: 
                p = bar.get_width()-0.02
            else: 
                p = bar.get_width()+0.005
            ax.text(p, bar.get_y()+0.15, str(round((n_importances[i]), 2)), fontsize=10, color='black')
    ax.set_title("Feature Importances - " + title, fontsize=20, loc='left', pad=30)
    ax.set_ylabel("Feature")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    plt.ylim(-1, n)
    fig.tight_layout()
    plt.grid()
    



# Comparing strategies
def Top_k_loss(x_train, y_train, x_val, y_val, feat, imp, k=15, metric=log_loss):
    model = rf_model(x_train, y_train, mode='C')
    loss_list = []
    n_imp = pd.Series(imp, index=feat).sort_values(ascending=False)[:k+1]
    for i in range(1, k+1):
        model_ = clone(model)
        features = n_imp.index[:i]
        model_.fit(x_train.loc[:, features], y_train)
        pred = model_.predict_proba(x_val.loc[:, features])
        loss = metric(y_val, pred)
        loss_list.append(loss)
    return loss_list


def compare_Top_k(data, target, k=10):
    train, val = train_val_split(data, 0.8)
    x_train, y_train = split_target(train, target)
    x_val, y_val = split_target(val, target)
    
    feat_spearman, imp_spearman = top_rank(data, target, method='spearman')
    loss_spearman = Top_k_loss(x_train, y_train, x_val, y_val, feat_spearman, imp_spearman, k=k)
    feat_pearson, imp_pearson = top_rank(data, target, method='pearson')
    loss_pearson = Top_k_loss(x_train, y_train, x_val, y_val, feat_pearson, imp_pearson, k=k)
    feat_kendall, imp_kendall = top_rank(data, target, method='kendall')
    loss_kendall = Top_k_loss(x_train, y_train, x_val, y_val, feat_kendall, imp_kendall, k=k)
    feat_pca, imp_pca = top_rank(data, target, method='PCA')
    loss_pca = Top_k_loss(x_train, y_train, x_val, y_val, feat_pca, imp_pca)
    imp_perm, feat_perm = permutation_importance(x_train, y_train, x_val, y_val, mode='R')
    loss_perm = Top_k_loss(x_train, y_train, x_val, y_val, feat_perm, imp_perm, k=k)
    imp_drop, feat_drop = dropcol_importances(x_train, y_train, x_val, y_val, mode='R')
    loss_drop = Top_k_loss(x_train, y_train, x_val, y_val, feat_drop, imp_drop, k=k)
    imp_shap, feat_shap = shap_importances(x_train, y_train, x_val, y_val)
    loss_shap = Top_k_loss(x_train, y_train, x_val, y_val, feat_shap, imp_shap, k=k)
    imp_mrmr, feat_mrmr = mRMR(data, target)
    loss_mrmr = Top_k_loss(x_train, y_train, x_val, y_val, feat_mrmr, imp_mrmr, k=k)

    fig = plt.figure(figsize=(15,15))
    ax = plt.axes()
    ax.grid(False)
    x, markers = range(1, k+1), ['o', '8', 's', 'p', '+', '*', 'h', 'v']
    plt.plot(x, loss_spearman, '#BA5645', marker=markers[0],  label='Spearman')
    plt.plot(x, loss_pearson, '#BA8949', marker=markers[1],  label='Pearson')
    plt.plot(x, loss_kendall, '#8DBA49', marker=markers[2],  label='Kendall')
    plt.plot(x, loss_pca, '#49A7BA', marker=markers[3],  label='PCA')
    plt.plot(x, loss_perm, '#6E49BA', marker=markers[4],  label='Permutation')
    plt.plot(x, loss_drop, '#BA49A0', marker=markers[5],  label='Drop Column')
    plt.plot(x, loss_shap, '#878784', marker=markers[6],  label='Shap')
    plt.plot(x, loss_mrmr, '#000000', marker=markers[7],  label='mRMR')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    ax.set_ylabel('Log Loss', fontsize=10)
    ax.set_xlabel('Top K selected features', fontsize=10)
    plt.show()


    
# Automatic feature selection
def auto_featSelection(data, target, mode='permutation', metric=log_loss):
    train, val = train_val_split(data, 0.8)
    x_train, y_train = split_target(train, target)
    x_val, y_val = split_target(val, target)
    model = rf_model(x_train, y_train, mode='C')
    model_ = clone(model)
    model_.fit(x_train, y_train)
    pred = model_.predict_proba(x_val)
    val_loss = metric(y_val, pred)
    
    # choose mode for featimp
    if mode == 'spearman':
        feat, imp = top_rank(data, target, method='spearman')
    elif mode == 'pearson':
        feat, imp = top_rank(data, target, method='pearson')
    elif mode == 'kendall':
        feat, imp = top_rank(data, target, method='kendall')
    elif mode == 'pca':
        feat, imp = top_rank(data, target, method='PCA')
    elif mode == 'permutation':
        imp, feat = permutation_importance(x_train, y_train, x_val, y_val, mode='R')
    elif mode == 'dropcol':
        imp, feat = dropcol_importances(x_train, y_train, x_val, y_val, mode='R')
    elif mode == 'shap':
        imp, feat = shap_importances(x_train, y_train, x_val, y_val)
    elif mode == 'mrmr':
        imp, feat = mRMR(data, target)
    else:
        print('Wrong mode name')
    
    val_loss_new = 0
    i = 0
    while True:
        i += 1
        drop_feat = feat[-i:]
        model_ = clone(model)
        x_train_drop = x_train.drop(columns=drop_feat)
        x_val_drop = x_val.drop(columns=drop_feat)
        model_.fit(x_train_drop, y_train)
        pred_new = model_.predict_proba(x_val_drop)
        val_loss_new = metric(y_val, pred_new)
        # if worse, use the previos one
        if val_loss_new > val_loss:
            if i == 1:
                return model_, []
            drop_feat = feat[-i+1:]
            model_ = clone(model)
            x_train_drop = x_train.drop(columns=drop_feat)
            x_val_drop = x_val.drop(columns=drop_feat)
            model_.fit(x_train_drop, y_train)
            break
        val_loss = val_loss_new
    return model_, drop_feat



# Variance and empirical p-values for feature importances
def feature_variance(data, target, mode='shap'):
    """
    Calculate standard deviation using booststraping
    """
    train, val = train_val_split(data, 0.8)
    x_train, y_train = split_target(train, target)
    n = 100
    imp_n = []
    for i in range(n):
        idx = np.random.choice(range(val.shape[0]), size=val.shape[0], replace=True)
        x_new, y_new = split_target(val.iloc[idx, :], target)
        if mode == 'shap':
            imp, _ = shap_importances(x_train, y_train, x_new, y_new)
        elif mode == 'permutation':
            imp, _ = permutation_importance(x_train, y_train, x_new, y_new)
        elif mode == 'dropcol':
            imp, _ = dropcol_importances(x_train, y_train, x_new, y_new)
        imp_n.append(imp)
    return np.std(np.array(imp_n), axis=0)


def feature_pvalue(data, target, mode='shap', metric=log_loss):
    train, val = train_val_split(data, 0.8)
    x_train, y_train = split_target(train, target)
    x_val, y_val = split_target(val, target)
    n = 100
    n_imp = []
    if mode == 'shap':
        baseline, feas = shap_importances(x_train, y_train, x_val, y_val)
    elif mode == 'permutation':
        baseline, feas = permutation_importance(x_train, y_train, x_val, y_val)
    elif mode == 'dropcol':
        baseline, feas = dropcol_importances(x_train, y_train, x_val, y_val)
    else:
        print('Wrong mode name')
        return
    baseline = baseline / np.sum(baseline)
    
    for i in range(n):
        idx = np.random.choice(range(val.shape[0]), size=val.shape[0], replace=True)
        x_new, y_new = split_target(val.iloc[idx, :], target)
        if mode == 'shap':
            imp, _ = shap_importances(x_train, y_train, x_new, y_new)
        elif mode == 'permutation':
            imp, _ = permutation_importance(x_train, y_train, x_new, y_new)
        elif mode == 'dropcol':
            imp, _ = dropcol_importances(x_train, y_train, x_new, y_new)
        imp = imp / np.sum(imp)
        n_imp.append(imp)
    diff = baseline - n_imp
    p_values = np.sum(diff <= 0, axis=0) / n
    return p_values, baseline, np.array(n_imp), feas


def pvalue_hist(p_values, baseline, imps, feas, k=0, size=(14,8), alpha=0.05):
    """
    Create a null distribution histogram for given top kth feature
    """
    list_plots = []
    fig, ax = plt.subplots(figsize=size)
    plt.hist(imps[:, k], bins='auto')
    ax.axvline(x=baseline[k], c='red')
    if p_values[k] < alpha:
        plt.title(f"Null Distributions Histogram for significant feature: {feas[k]} with p-value: {p_values[k]}")
    else:
        plt.title(f"Null Distributions Histogram for insignificant feature: {feas[k]} with p-value: {p_values[k]}")
    plt.show()