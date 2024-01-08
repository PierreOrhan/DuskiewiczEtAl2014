from uncertainties import unumpy
import  numpy as np

def linear_model_with_error(x,y,sigma_x,sigma_y):
    x = unumpy.uarray(np.log(x ** 2), sigma_x)
    y = unumpy.uarray(np.log(y ** 2), sigma_y)
    sx = unumpy.sqrt(unumpy.exp(x))
    sy = unumpy.sqrt(unumpy.exp(y))
    alpha = np.sum(np.transpose(sx) * sy) / np.sum(np.transpose(sx) * sx)
    pred = alpha * sx
    SSres = np.sum((pred - sy) ** 2)
    SStot = np.sum((sy - np.mean(sy)) ** 2)
    model_r2 = 1 - (SSres / SStot)
    return model_r2,alpha


def _linearmodelR2(x,y,sigma_x,alpha):
    x = unumpy.uarray(np.log(x ** 2), sigma_x)
    y = np.log(y ** 2)
    sx = unumpy.sqrt(unumpy.exp(x))
    sy = unumpy.sqrt(unumpy.exp(y))
    pred = alpha * sx
    SSres = np.sum((pred - sy) ** 2)
    SStot = np.sum((sy - np.mean(sy)) ** 2)
    model_r2 = 1 - (SSres / SStot)
    return model_r2


def linear_model_with_error_2d(x1,x2,y,sigma_x1,sigma_x2,sigma_y):
    x1 = unumpy.uarray(np.log(x1 ** 2), sigma_x1)
    x2 = unumpy.uarray(np.log(x2 ** 2), sigma_x2)
    x = np.stack([x1,x2],axis=1)
    y = unumpy.uarray(np.log(y ** 2), sigma_y)
    sx = unumpy.sqrt(unumpy.exp(x))
    sy = unumpy.sqrt(unumpy.exp(y))
    beta = unumpy.ulinalg.inv( np.transpose(sx) @ sx)
    alpha = beta @(np.transpose(sx) @ sy)
    pred = alpha @ np.transpose(sx)
    SSres = np.sum((pred - sy) ** 2)
    SStot = np.sum((sy - np.mean(sy)) ** 2)
    model_r2 = 1 - (SSres / SStot)
    return model_r2,alpha

def _2dlinearmodelR2(x1,x2,y,sigma_x1,sigma_x2,sigma_y,alpha):
    x1 = unumpy.uarray(np.log(x1 ** 2), sigma_x1)
    x2 = unumpy.uarray(np.log(x2 ** 2), sigma_x2)
    x = np.stack([x1,x2],axis=1)
    y = unumpy.uarray(np.log(y ** 2), sigma_y)
    sx = unumpy.sqrt(unumpy.exp(x))
    sy = unumpy.sqrt(unumpy.exp(y))
    pred = alpha @ np.transpose(sx)
    SSres = np.sum((pred - sy) ** 2)
    SStot = np.sum((sy - np.mean(sy)) ** 2)
    model_r2 = 1 - (SSres / SStot)
    return model_r2

def _2dlinearmodelR2_noUncertaininty(X,y,alpha):
    pred = alpha @ np.transpose(X)
    SSres = np.sum((pred - y) ** 2)
    SStot = np.sum((y - np.mean(y)) ** 2)
    model_r2 = 1 - (SSres / SStot)
    return model_r2
