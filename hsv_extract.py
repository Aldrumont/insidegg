import cv2
import numpy as np

# Função callback para os sliders
def on_trackbar(val):
    pass

# Função para ajustar os limites de HSV e aplicar a máscara
def apply_mask(image, hsv_image):
    # Pegando os valores dos sliders
    h_min = cv2.getTrackbarPos('H Min', 'Segmented Image')
    s_min = cv2.getTrackbarPos('S Min', 'Segmented Image')
    v_min = cv2.getTrackbarPos('V Min', 'Segmented Image')
    h_max = cv2.getTrackbarPos('H Max', 'Segmented Image')
    s_max = cv2.getTrackbarPos('S Max', 'Segmented Image')
    v_max = cv2.getTrackbarPos('V Max', 'Segmented Image')

    # Convertendo os valores para a escala de OpenCV
    lower_bound = np.array([h_min // 2, s_min * 255 // 360, v_min * 255 // 360])
    upper_bound = np.array([h_max // 2, s_max * 255 // 360, v_max * 255 // 360])

    # Aplicando a máscara
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    
    return segmented_image

# Carregando a imagem
image = cv2.imread('frame_00093.jpg')  # Substitua 'your_image.jpg' pelo caminho da sua imagem
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Criando a janela e os sliders
cv2.namedWindow('Segmented Image')
cv2.createTrackbar('H Min', 'Segmented Image', 0, 360, on_trackbar)
cv2.createTrackbar('S Min', 'Segmented Image', 0, 360, on_trackbar)
cv2.createTrackbar('V Min', 'Segmented Image', 0, 360, on_trackbar)
cv2.createTrackbar('H Max', 'Segmented Image', 360, 360, on_trackbar)
cv2.createTrackbar('S Max', 'Segmented Image', 360, 360, on_trackbar)
cv2.createTrackbar('V Max', 'Segmented Image', 360, 360, on_trackbar)

while True:
    # Atualizando a imagem segmentada
    segmented_image = apply_mask(image, hsv_image)

    # Mostrando a imagem
    cv2.imshow('Segmented Image', segmented_image)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
