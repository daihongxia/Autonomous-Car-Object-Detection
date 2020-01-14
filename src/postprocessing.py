import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from preprocessing import coords2str,rotate

def _obtain_all_points(train):
    points_df = pd.DataFrame()
    for col in ['x', 'y', 'z', 'yaw', 'pitch', 'roll']:
        arr = []
        for ps in train['PredictionString']:
            coords = str2coords(ps)
            arr += [c[col] for c in coords]
        points_df[col] = arr
    return points_df

def _obtain_xzy_slope(points_df):
    # Will use this model later
    xzy_slope = LinearRegression()
    X = points_df[['x', 'z']]
    y = points_df['y']
    xzy_slope.fit(X, y)
    #print('MAE with x:', mean_absolute_error(y, xzy_slope.predict(X)))

    #print('\ndy/dx = {:.3f}\ndy/dz = {:.3f}'.format(*xzy_slope.coef_))
    return xzy_slope

def convert_3d_to_2d(x, y, z, fx = 2304.5479, fy = 2305.8757, cx = 1686.2379, cy = 1354.9849):
    # stolen from https://www.kaggle.com/theshockwaverider/eda-visualization-baseline
    return x * fx / z + cx, y * fy / z + cy

def optimize_xy(r, c, x0, y0, z0, img_w=1600, img_h=500, model_scale=8, xzy_slope=None, flipped=False):
    def distance_fn(xyz):
        IMG_SHAPE = (2710, 3384, 3)
        x, y, z = xyz
        xx = -x if flipped else x
        x, y = convert_3d_to_2d(x, y, z)
        y, x = x, y
        x = (x - IMG_SHAPE[0] // 2) * img_h / (IMG_SHAPE[0] // 2) / model_scale
        y = (y + IMG_SHAPE[1] // 6) * img_w / (IMG_SHAPE[1] * 4 / 3) / model_scale
        if xzy_slope is not None:
            slope_err = (xzy_slope.predict([[xx,z]])[0] - y)**2
            return max(0.2, (x-r)**2 + (y-c)**2) + max(0.4, slope_err)
        return (x-r)**2 + (y-c)**2
    
    res = minimize(distance_fn, [x0, y0, z0], method='Powell')
    x_new, y_new, z_new = res.x
    return x_new, y_new, z_new

def clear_duplicates(coords, dist_thres=2):
    for c1 in coords:
        xyz1 = np.array([c1['x'], c1['y'], c1['z']])
        for c2 in coords:
            xyz2 = np.array([c2['x'], c2['y'], c2['z']])
            distance = np.sqrt(((xyz1 - xyz2)**2).sum())
            if distance < dist_thres:
                if c1['confidence'] < c2['confidence']:
                    c1['confidence'] = -1
    return [c for c in coords if c['confidence'] > 0]

def find_peak(a):
    ch1 = np.concatenate([np.array([[False]*a.shape[0]]).T, (a[:,1:]-a[:,:-1])>=0],axis=1)
    ch2 = np.concatenate([(a[:,:-1]-a[:,1:])>=0, np.array([[False]*a.shape[0]]).T],axis=1)
    ch3 = np.concatenate([(a[:-1,:]-a[1:,])>=0, np.array([[False]*a.shape[1]])],axis=0)
    ch4 = np.concatenate([np.array([[False]*a.shape[1]]),(a[1:,:]-a[:-1,])>=0],axis=0)
    return ch1*ch2*ch3*ch4

def _regr_back(regr_dict):
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] * 100
    regr_dict['roll'] = rotate(regr_dict['roll'], -np.pi)
    
    pitch_sin = regr_dict['pitch_sin'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    pitch_cos = regr_dict['pitch_cos'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    regr_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)
    return regr_dict

def extract_coords(prediction, threshold=0, flipped=False):
    logits = prediction[0]
    regr_output = prediction[1:]
    
    mask1 = (logits>threshold)
    mask2 = find_peak(logits)
    mask = mask1*mask2
    
    points = np.argwhere(mask)
    
    col_names = sorted(['x', 'y', 'z', 'yaw', 'pitch_sin', 'pitch_cos', 'roll'])
    coords = []
    for r, c in points:
        regr_dict = dict(zip(col_names, regr_output[:, r, c]))
        coords.append(_regr_back(regr_dict))
        coords[-1]['confidence'] = 1 / (1 + np.exp(-logits[r, c]))
        coords[-1]['x'], coords[-1]['y'], coords[-1]['z'] = \
                optimize_xy(r, c,
                            coords[-1]['x'],
                            coords[-1]['y'],
                            coords[-1]['z'], 
                            img_w=1600, img_h=500, model_scale=8,xzy_slope=None,
                            flipped=flipped)
    coords = clear_duplicates(coords)
    return coords
