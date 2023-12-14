import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
from numba import jit
import os
from matplotlib.widgets import Slider

# ------------------------ Funciones comunes --------------------------
# Leer imagen
def read_image(path, eigth_bits = True, one_channel = True):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if eigth_bits:
        if img.dtype == 'uint16':
            img = (img/256).astype('uint8')
    if one_channel:
        if len(img.shape) == 3:
            img = img[:,:,0]
    return img

# Muestra la imagen en una ventana
def show_image(img, scale_percent = 75, nombre="Imagen"):
  width = int(img.shape[1] * scale_percent / 100)
  height = int(img.shape[0] * scale_percent / 100)
  dim = (width, height)

  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)  # resize
  cv2.imshow(nombre, resized)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

# Guardar imagen y añadir texto al final del nombre
def save_img(img, img_path, legend=''):
    dir_name, ext = os.path.splitext(img_path)

    tiff = 0
    if ext == ".tif" or ext == ".tiff":
        tiff = 1
    try:
      cv2.imwrite(f"{dir_name}_{legend}{ext}", img, [cv2.IMWRITE_EXR_COMPRESSION_NO, 1, cv2.IMWRITE_TIFF_COMPRESSION, tiff])
    except:
       print("Cannot save image")

# Elimina las n_rows filas superiores de la imagen
def delete_top(image, n_rows):
  return np.delete(image, range(0, n_rows) , 0)

# Elimina las n_rows filas inferiores de la imagen
def delete_bottom(image, n_rows):
  N = image.shape[0]
  return np.delete(image, range(N-n_rows, N) , 0)

# Elimina las n_cols columnas a la izquierda de la imagen
def delete_left(image, n_cols):
  return np.delete(image, range(0, n_cols) , 1)

# Elimina las n_cols columnas a la izquierda de la imagen
def delete_right(image, n_cols):
  N = image.shape[1]
  return np.delete(image, range(N-n_cols, N) , 1)

# Analiza brevemente la distribución provista
def measure_distribution(measures, plot=True):
  """Grafica la distribución de mediciones y calcula
     su media e incertidumbre.

        Parametros
        ----------
        measures : numpy.array, vector de mediciones.

        plot : bool, Define si se grafica o no. Default: True.

        Retorna
        ------
        (mean, u) : tuple, Tupla con la media y la incertidumbre
                  tipo A obtenida de la distribución.
  """
  count = len(measures)
  mean = measures.mean()  #media
  u = measures.std()/np.sqrt(count) #Incertidumbre de la media

  if plot:
    d = np.diff(np.unique(measures)).min()
    left_of_first_bin = measures.min() - float(d)/2
    right_of_last_bin = measures.max() + float(d)/2
    plt.hist(measures, np.arange(left_of_first_bin, right_of_last_bin + d, d))
    plt.show()
    print(f"{count} filas con media {mean} ({u}) px\n")

  return (mean, u)


# --------------- Funciones de "Medición" de pixeles -------------------
# Primer pixel de la fila (en blanco)
# Optimizada con numba
@jit(nopython=True)
def first_pixel(row):
    for i in range(len(row)):
        if 255 == row[i]:
            return i
    return None

# Ultimo pixel de la fila (en blanco)
# Optimizada con numba
@jit(nopython=True)
def last_pixel(row):
  i = first_pixel(np.flip(row)) # Indice "desde atras" / Optimizar
  if i is not None:
    return len(row) - i - 1
  else:
    return None

def direct_distance(mat, min_dif=10, outlier_percent = 2, plot=True):
  """Calcula la distancia entre los bordes verticales
    paralelos de una imagen en blanco y negro, fila por fila.

        Parametros
        ----------
        mat : 2-D numpy.array, matriz que contiene el valor
        de los pixeles de la imagen

        min_dif : int, Mínima separación de los bordes (en px)
        para considerar correcta una medición. Default: 10.

        outlier_percent : Float, Minimo porcentaje respecto a la
        moda para considerar una medición como outlier (y descartarla).
        Default: 2 %

        plot: bool, Define si se imprimen o no mensajes en consola.
        Default: True

        Retorna
        ------
        p_diams : 1-D numpy.array, Array con las mediciones obtenidas
        que cumplen con los parámetros.
  """

  #Validaciones
  try:
    assert mat.ndim == 2
  except:
    print("mat debe tener 2 dimensiones (matriz B y N)")
    return
  try:
    assert type(outlier_percent)==float or type(outlier_percent)== int
  except:
    print("outlier_percents debe ser de tipo float o int")
    return
  try:
    assert type(min_dif)==int
  except:
    print("min_dif debe ser int")
    return

  p_diams = []
  for row in mat:       # Recorre cada fila y mide la distancia entre bordes
    first = first_pixel(row)
    last = last_pixel(row)
    if first and last:
      diam = last - first
      if diam > min_dif:
        p_diams.append(diam)

  p_diams = np.array(p_diams) # Array de mediciones

  # Procesamiento de outliers
  moda = np.bincount(p_diams).argmax()
  max = (1+outlier_percent/100)*moda
  min = (1-outlier_percent/100)*moda
  outliers =  np.where( (p_diams > max) | (p_diams < min) )

  if plot:
    print(f"Media inicial: {p_diams.mean()}")
    print(f"Moda: {moda}")

  p_diams = np.delete(p_diams, outliers)

  if plot:
    print(f"Media final: {p_diams.mean()}\n")
  return p_diams


# ------------------- Métodos de Detección de bordes -----------------

# Clase para definir los valores de umbral de las funciones de detección.
# Al inicializar una instancia debe pasarse la imagen, la función de detección,
# si se utilizan dos valores de umbral (uno si False) y  el valor de escala
# para visualizar los bordes en pantalla.
class DefineThreshold():
  def __init__(self, img, fun, two_params=True, th_max = None):
        self.img = img
        self.fun = fun
        self.two_params = two_params
        if th_max:
           self.th_max = th_max
        else:
          self.th_max = img.max()
        self.th1 = self.th_max//2
        self.th2 = self.th_max
        self.winName = "Deteccion de bordes"
        self.pzw = PanZoomWindow(self.img, self.img, self.winName)

  def __update_edges(self, edge, name):
    if self.pzw.original:
      self.pzw.aux_img = edge
    else:
      self.pzw.img = edge
    self.pzw.show()

  def __find_edges(self, thresh1=100, thresh2=255, **kwargs):
      try:
        self.th1 = cv2.getTrackbarPos('Thresh 1', self.winName)
        if self.two_params:
          self.th2 = cv2.getTrackbarPos('Thresh 2', self.winName)
      except:
        self.th1 = self.th_max//2
        self.th2 = self.th_max
      if self.two_params:
        edge = self.fun(self.img, self.th1, self.th2, **kwargs)
      else:
         edge = self.fun(self.img, self.th1, **kwargs)
      self.__update_edges(edge, self.winName)

  # Llamar para obtener umbrales mediante GUI.
  def get_th(self, **kwargs): 
    self.pzw.show()
    cv2.namedWindow(self.winName)
    cv2.createTrackbar('Thresh 1', self.winName, self.th1, self.th_max, self.__find_edges)
    if self.two_params:
      cv2.createTrackbar('Thresh 2', self.winName, self.th2, self.th_max, self.__find_edges)
  
    self.__find_edges(self.th1, self.th2, **kwargs)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return (self.th1, self.th2)
  

class EdgesViewer():
    def __init__(self, orig, edges, first_edge = True, N_points=75):
        plt.close('all')

        self.orig_blur = cv2.GaussianBlur(orig, (3, 3), sigmaX=0, sigmaY=0)  # Blurea la imagen (mejor deteccion)
        if first_edge:
            self.ix_edges = np.array([first_pixel(row) for row in edges], dtype=np.int32)
            np.nan_to_num(self.ix_edges, copy=False)
        else:
            self.ix_edges = np.array([last_pixel(row) for row in edges], dtype=np.int32)
            np.nan_to_num(self.ix_edges, copy=False)
        center = int(np.mean(self.ix_edges))
        self.xmin = center - N_points//2
        self.xmax = center + N_points//2

    def plot(self):
        fig, ax = plt.subplots(figsize=(10,5))
        ax.set_ylim(self.orig_blur.min(), self.orig_blur.max() + 1)
        line, = ax.plot(self.orig_blur[0,self.xmin:self.xmax,...])
        edge = ax.axvline(self.ix_edges[0] - self.xmin, c='r')
        fig.tight_layout()
        fig.subplots_adjust(left=0.05, bottom=0.2, right=0.98, top=0.99)

        axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        self.slider = Slider(
            ax=axfreq,
            valstep = 1,
            label='Número de Fila',
            valmin=0,
            valmax=self.orig_blur.shape[0]-1,
            valinit=0,
        )

        def update(val):
            # Actualizar la posición de la línea vertical
            edge.set_xdata([self.ix_edges[val] - self.xmin])
            # Actualizar la línea del gráfico principal
            line.set_ydata(self.orig_blur[val, self.xmin:self.xmax, ...])
            # Redibujar la figura
            fig.canvas.draw_idle()

        self.slider.on_changed(update)
        ax.grid()
        plt.show()

# Devuelve imagen en blanco y negro de acuerdo a los valores de umbral
# Este método es simple pero garantiza no incluir información "exterior" al pixel
# (a diferencia de otros métodos, no importa el valor del pixel aledaño)
def Thresholding(mat, thresh=200, max_val=255):
    '''Convierte la imagen de acuerdo al valor del del pixel en escala de grises
      respecto al umbral.
      Devuelve bordes detectados (imagen de 8 bits)'''
    if len(mat.shape) == 3:
      mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)   #Convierte a grises
    _,th = cv2.threshold(mat, thresh, max_val, cv2.THRESH_BINARY)   #Aplica umbral
    if th.dtype == 'uint16':
            th = (th/256).astype('uint8')
    return th

# Devuelve bordes detectados mediante Canny
def Canny_Edges(mat, th1=100, th2=200, L2gradient=True, blur_size=3):
    '''Encuentra bordes en la imagen de entrada (mat) y los marca en los bordes del mapa
       de salida utilizando el algoritmo Canny.
       El valor más pequeño entre th1 y th2 se utiliza para la vinculación de bordes.
       El valor más grande se utiliza para encontrar segmentos iniciales de bordes fuertes.
       L2gradient: flag, indica si se utiliza norma exacta o no
       blur_size: tamaño del kernel para el filtro gaussiano'''
    img_blur = cv2.GaussianBlur(mat, (blur_size, blur_size), sigmaX=0, sigmaY=0)  # Blurea la imagen (mejor deteccion)
    edges = cv2.Canny(img_blur, th1, th2, L2gradient=L2gradient) # Canny Edge Detection
    return edges

# Aplica filtro laplaciano (pasaalto) y devuelve
# imagen en blanco y negro de acuerdo a los valores de umbral
# FUNCIONA PEOR QUE LOS OTROS MÉTODOS
def Laplacian(mat, thresh=127, max_val=255):
    '''Aplica filtro laplaciano (pasaalto) y convierte la imagen de acuerdo
      al valor del del pixel en escala de grises respecto al umbral.'''
    img_blur = cv2.GaussianBlur(mat, (3,3), sigmaX=0, sigmaY=0)  # Blurea la imagen (mejor deteccion)
    if len(mat.shape) == 3:
      img_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)   #Convierte a grises
    dst = cv2.Laplacian(img_blur, cv2.CV_64F, ksize=3)
    abs_dst = cv2.convertScaleAbs(dst)
    _,th = cv2.threshold(abs_dst, thresh, max_val, cv2.THRESH_BINARY)   #Aplica umbral
    return th

# Encuentra el "codo" (primer punto de cambio) en un array 1D
def findKnee(y, N = 30, direction='inc'):
    """Encuentra el "codo" (primer punto de cambio) en un array 1D.
      Conviene utilizar array previamente suavizado (blur).

        Parametros
        ----------
        y : 1-D numpy.array, vector de datos.

        direction : str {'inc', 'dec'}, Define si se busca en el flanco
        ascendente ('inc') o descendente ('dec'). Default: 'inc'.

        N : int, Número de puntos alrededor del flanco para buscar el codo. 
        En general, cuanto mayor, más al inicio resulta el punto encontrado.
        Default: 30

        Retorna
        ------
        Índice del codo encontrado.
    """
    g = np.gradient(y)
    L = len(y)
    ix_bot = None
    ix_top = None
    
    if direction=='inc':
        ix_g = g.argmax()           # índice de mayor gradiente (flanco asc.)
        ix1 = ix_g-N
        ix2 = ix_g+N
        if ix1 > 0 and ix2<L:
          yy = y[(ix1):(ix2)]     # Slice cerca del flanco
          ix_top = yy.argmax() + ix1 + 1   # Limite superior (en el máximo valor)
          ix_bot = ix_top - 2*N
          ix_bot = ix_bot if ix_bot>0 else 0
    elif direction=='dec':
        ix_g = g.argmin()           # índice de mayor gradiente (flanco desc.)
        ix1 = ix_g-N
        ix2 = ix_g+N
        if ix1 > 0 and ix2<L:
          yy = y[(ix1):(ix2)]     # Slice cerca del flanco
          ix_bot = yy.argmax() + ix1  # Limite inferior (en el máximo valor)
          ix_top = ix_bot + 2*N
          ix_top = ix_top if ix_top<L else L-1

    else:
       print("Direction only can be: ('inc', 'dec')")
    
    if ix_bot and ix_top:
      yy = y[ix_bot:ix_top]       # x_bot es el nuevo cero
      xx = np.arange(len(yy))
    else:
       return None

    # Genera una recta que une el primer y último punto
    m = (int(yy[-1]) - int(yy[0])) / (len(yy))
    c = yy[0]
    line = m * xx + c

    # Obtiene los residuos
    res = yy - line
    ix_res_min = np.argmin(res) # índice de mayor residuo (negativo)

    return ix_res_min + ix_bot  # Devuelve índice respecto al cero del slice

def Knee_Detection(mat, N):
    '''Detecta bordes fila por fila usando el método del codo
    (primer punto de cambio).
    N: Entorno de trabajo respecto al flanco.'''
    edges = np.zeros((mat.shape[0], mat.shape[1]))
    if N < 2:
       return edges
    
    mat = cv2.GaussianBlur(mat, (3,3), sigmaX=0, sigmaY=0)  # Blurea la imagen (mejor deteccion)
    if len(mat.shape) == 3:
      mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)   #Convierte a grises

    for n, row in enumerate(mat):
      ix_1 = findKnee(row, N, 'inc')
      ix_2 = findKnee(row, N, 'dec')
      if ix_1:
        edges[n, ix_1] = 255
      if ix_2:
        edges[n, ix_2] = 255

    return edges

# -------------------------------------------------------------------------------#

class PanZoomWindow(object):
    """ Controls an OpenCV window. Registers a mouse listener so that:
        1. left-dragging up/down zooms in/out
        2. left-clicking re-centers
        3. trackbars scroll vertically and horizontally 
    You can open multiple windows at once if you specify different window names.
    You can pass in an onRigthClickFunction, and when the user left-clicks, this 
    will call onRigthClickFunction(y,x)"""
    def __init__(self, img, img2, windowName = 'PanZoomWindow', onRigthClickFunction = None):
        self.WINDOW_NAME = windowName
        self.H_TRACKBAR_NAME = 'x'
        self.V_TRACKBAR_NAME = 'y'
        self.img = img
        self.aux_img = img2
        self.original = True
        self.onRigthClickFunction = onRigthClickFunction
        self.TRACKBAR_TICKS = 1000
        self.panAndZoomState = PanAndZoomState(img.shape, self)
        self.lButtonDownLoc = None
        self.mButtonDownLoc = None
        self.rButtonDownLoc = None
        self.first_show = True

    def show(self):
        if self.first_show:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.WINDOW_NAME, self.onMouse)
            cv2.createTrackbar(self.H_TRACKBAR_NAME, self.WINDOW_NAME, 0, self.TRACKBAR_TICKS, self.onHTrackbarMove)
            cv2.createTrackbar(self.V_TRACKBAR_NAME, self.WINDOW_NAME, 0, self.TRACKBAR_TICKS, self.onVTrackbarMove)
            self.first_show = False
        else:
            if cv2.getWindowProperty(self.WINDOW_NAME, 0) >= 0:     # If window is still open
                cv2.imshow(self.WINDOW_NAME, self.img if self.original else self.aux_img)
        self.redrawImage()

    def onMouse(self,event, x,y,_ignore1,_ignore2):
        """ Responds to mouse events within the window. 
        The x and y are pixel coordinates in the image currently being displayed.
        If the user has zoomed in, the image being displayed is a sub-region, so you'll need to
        add self.panAndZoomState.ul to get the coordinates in the full image."""
        if event == cv2.EVENT_MOUSEMOVE:
            return
        elif event == cv2.EVENT_LBUTTONDOWN:
            #record where the user started to right-drag
            self.mButtonDownLoc = np.array([y,x])
        elif event == cv2.EVENT_LBUTTONUP and self.mButtonDownLoc is not None:
            #the user just finished right-dragging
            dy = y - self.mButtonDownLoc[0]
            pixelsPerDoubling = 0.2*self.panAndZoomState.shape[0] #lower = zoom more
            changeFactor = (1.0+abs(dy)/pixelsPerDoubling)
            changeFactor = min(max(1.0,changeFactor),5.0)
            if changeFactor < 1.05:
                dy = 0 #this was a click, not a draw. So don't zoom, just re-center.
            if dy > 0: #moved down, so zoom out.
                zoomInFactor = 1.0/changeFactor
            else:
                zoomInFactor = changeFactor
#            print("zoomFactor: %s"%zoomFactor)
            self.panAndZoomState.zoom(self.mButtonDownLoc[0], self.mButtonDownLoc[1], zoomInFactor)
        elif event == cv2.EVENT_RBUTTONDOWN:
            #the user pressed the rigth button. 
            coordsInDisplayedImage = np.array([y,x])
            if np.any(coordsInDisplayedImage < 0) or np.any(coordsInDisplayedImage > self.panAndZoomState.shape[:2]):
                print("you clicked outside the image area")
            else:
                self.img, self.aux_img = self.aux_img, self.img
                self.original = not self.original
                self.redrawImage()

    def onVTrackbarMove(self,tickPosition):
        self.panAndZoomState.setYFractionOffset(float(tickPosition)/self.TRACKBAR_TICKS)

    def onHTrackbarMove(self,tickPosition):
        self.panAndZoomState.setXFractionOffset(float(tickPosition)/self.TRACKBAR_TICKS)

    def redrawImage(self):
        pzs = self.panAndZoomState
        cv2.imshow(self.WINDOW_NAME, self.img[pzs.ul[0]:pzs.ul[0]+pzs.shape[0], pzs.ul[1]:pzs.ul[1]+pzs.shape[1]])

# -------------------------------------------------------------------------------#

class PanAndZoomState(object):
    """ Tracks the currently-shown rectangle of the image.
    Does the math to adjust this rectangle to pan and zoom."""
    MIN_SHAPE = np.array([50,50])
    def __init__(self, imShape, parentWindow):
        self.ul = np.array([0,0]) #upper left of the zoomed rectangle (expressed as y,x)
        self.imShape = np.array(imShape[0:2])
        self.shape = self.imShape #current dimensions of rectangle
        self.parentWindow = parentWindow

    def zoom(self,relativeCy,relativeCx,zoomInFactor):
        self.shape = (self.shape.astype(float)/zoomInFactor).astype(int)
        #expands the view to a square shape if possible. (I don't know how to get the actual window aspect ratio)
        self.shape[:] = np.max(self.shape) 
        self.shape = np.maximum(PanAndZoomState.MIN_SHAPE,self.shape) #prevent zooming in too far
        c = self.ul+np.array([relativeCy,relativeCx])
        self.ul = (c-self.shape/2).astype(int)
        self._fixBoundsAndDraw()

    def _fixBoundsAndDraw(self):
        """ Ensures we didn't scroll/zoom outside the image. 
        Then draws the currently-shown rectangle of the image."""
        self.ul = np.maximum(0,np.minimum(self.ul, self.imShape-self.shape))
        self.shape = np.minimum(np.maximum(PanAndZoomState.MIN_SHAPE,self.shape), self.imShape-self.ul)
        yFraction = float(self.ul[0])/max(1,self.imShape[0]-self.shape[0])
        xFraction = float(self.ul[1])/max(1,self.imShape[1]-self.shape[1])
        cv2.setTrackbarPos(self.parentWindow.H_TRACKBAR_NAME, self.parentWindow.WINDOW_NAME,int(xFraction*self.parentWindow.TRACKBAR_TICKS))
        cv2.setTrackbarPos(self.parentWindow.V_TRACKBAR_NAME, self.parentWindow.WINDOW_NAME,int(yFraction*self.parentWindow.TRACKBAR_TICKS))
        self.parentWindow.redrawImage()
    def setYAbsoluteOffset(self,yPixel):
        self.ul[0] = min(max(0,yPixel), self.imShape[0]-self.shape[0])
        self._fixBoundsAndDraw()
    def setXAbsoluteOffset(self,xPixel):
        self.ul[1] = min(max(0,xPixel), self.imShape[1]-self.shape[1])
        self._fixBoundsAndDraw()
    def setYFractionOffset(self,fraction):
        """ pans so the upper-left zoomed rectange is "fraction" of the way down the image."""
        self.ul[0] = int(round((self.imShape[0]-self.shape[0])*fraction))
        self._fixBoundsAndDraw()
    def setXFractionOffset(self,fraction):
        """ pans so the upper-left zoomed rectange is "fraction" of the way right on the image."""
        self.ul[1] = int(round((self.imShape[1]-self.shape[1])*fraction))
        self._fixBoundsAndDraw()


# ---------------- Función para rotar y enderezar ---------------------
# Basado en las funciones de Andri & Magnus Hoff (en Stack Overflow)

def rotate_image(mat, angle):
    """
    Gira una imagen OpenCV2/NumPy sobre su centro en el ángulo dado
    (en grados). La imagen devuelta será lo suficientemente grande como para
    contener toda la imagen nueva, con fondo negro.
    """
    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), flags=cv2.INTER_CUBIC)

    return rotated_mat

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

# Funciones para cortar (Crop) la imagen rotada
def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

# Combina las funciones de rotar, calcular el mayor rectángulo y cortar
def rotate_and_cut(image, degrees):
    if degrees == 0:
       return image
    image_height, image_width = image.shape[0:2]
    image_rotated = rotate_image(image, degrees)
    max_size = largest_rotated_rect(image_width, image_height, math.radians(degrees))
    img = crop_around_center(image_rotated, *max_size)
    return img


# ------------ Funciones para ajustar el ángulo de rotación -------------

def ridge_grados(grados, results, orden, ax, alpha=0, label="y"):
        '''
        Realiza un ajuste polinomial entre el vector de grados
        y el de resultados. Primero se generan los parámetros polinomiales
        de entrada y se ajusta una función mediante Ridge (Regresión Lineal +
        regularización). Por último busca el mínimo de la función ajustada
        (se busca el valor de grados que arroja el mínimo valor de results).

        Parametros
        ----------
        grados : np.array, vector con los grados.

        results : np.array, vector con resultados (parámetro a minimizar).

        orden : int, Orden del polinomio generado.

        ax : Ejes actuales para graficar.

        alpha : float (positivo) Factor de regularización (Si alpha = 0
                se obtiene una Regresión lineal clásica).
                Default: 0.

        label : string, Label del eje Y del gráfico (results).
                Default: 'y'.

        Retorna
        ------
        best_aj : float, mejor ángulo encontrado (aquel que minimiza results).

        score : float, score resultante del ajuste.
        '''
        pipe = Pipeline([
                ('poly_features', PolynomialFeatures(orden)),
                ('regressor', Ridge(fit_intercept=False, alpha=alpha) )])

        pipe.fit(grados.reshape(-1,1), results)  # Ajuste polinomial (de grados y results)

        outliers = []   # Pese a la regularización, el ajuste es sensible a grandes outliers
        for i, g in enumerate(grados):
           pred = pipe.predict(g.reshape(1, -1))
           err = abs(pred - results[i])/pred
           if err > 0.85:     # Diferencia mayor a 85% (outlier grande)
              outliers.append(i)
        grados = np.delete(grados, outliers)    # Elimina outliers
        results = np.delete(results, outliers)

        # Recalcula
        pipe = Pipeline([
                ('poly_features', PolynomialFeatures(orden)),
                ('regressor', Ridge(fit_intercept=False, alpha=alpha) )])
        pipe.fit(grados.reshape(-1,1), results)
        score = pipe.score(grados.reshape(-1,1), results)

        ax.scatter(grados, results)
        ax.grid()
        ax.set_xlabel("Angulo de rotación (°)")
        ax.set_ylabel(label)
        g_plot = np.linspace(min(grados), max(grados), 200)
        m_plot = pipe.predict(g_plot.reshape(-1,1))
        ax.plot(g_plot, m_plot, c='r')  # Nota: Normalmente debe observarse una parabola creciente

        ix_min = results.argmin()   # Mínimo medido (provisto en results)
        best_med = grados[ix_min]

        # El mínimo de la función se puede aproximar directamente (si se tienen muchos puntos)
        ix_min1 = m_plot.argmin()
        best_aj = g_plot[ix_min1]

        # Hallar mínimo de la función ajustada con minimize()
        coef = pipe.named_steps['regressor'].coef_  # Coeficientes obtenidos del ajuste
        def func(x):
                res = 0
                for i in range(orden+1):
                        res += coef[i] * x**i
                return res
        best = minimize(func, x0=0)
        if best.success:
                best_aj = best.x[0]

        print(label)
        print(f"Mejor angulo medido: {round(best_med, 3)}°")
        print(f"Mejor angulo ajustado: {round(best_aj, 3)}°. Score={round(score, 3)}\n")

        return best_aj, score


def align(mat, degrees, fun, **kwargs):
    '''
      Función para elección del ángulo (empíricamente).
      Válido para ángulos pequeños (pueden perderse los bordes al cortar, verificar).
      Parametros
      ----------
      mat: np.array, matriz con la imagen.

      degrees : np.array, vector con los grados a probar.

      Retorna
      ------
      ang_ajustado : float, ángulo encontrado en el ajuste de cada lado (pesado según
      el score del ajuste).
    '''
    results2 = []

    for g in degrees:   # Verifica uno a uno los ángulos
        img = rotate_and_cut(mat, g)  # Rota la imagen g grados
        edges = fun(img, **kwargs)  # Detección de bordes
        # Mide todos los primeros y últimos pixeles
        firsts = []
        lasts = []
        for row in edges:
            f = first_pixel(row)
            l = last_pixel(row)
            if f:
                firsts.append(f)
            if l:
                lasts.append(l)
        results2.append( [np.std(firsts), np.std(lasts)])

    results2 = np.array(results2)

    # Ajuste y visualización
    fig, ax = plt.subplots(2)
    fig.tight_layout()
    angulo1, score1 = ridge_grados(degrees, results2[:,0], 2, ax[0], label="Desvío primer pixel (px)")
    angulo2, score2 = ridge_grados(degrees, results2[:,1], 2, ax[1], label="Desvío último pixel (px)")
    ang_ajustado = (score1*angulo1 + score2*angulo2)/(score1 + score2)
    print("Ajuste entre ambos lados:", round(ang_ajustado, 3), '°')
    return ang_ajustado

