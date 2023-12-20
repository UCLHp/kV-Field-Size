# Gui
import PySimpleGUI as sg
import sys

# Data analysis
import numpy as np

# Loading images
import cv2
import glob
import os

# Results
import pandas as pd
import matplotlib.pyplot as plt

# Report
from fpdf import FPDF

########################################################################################################################

#############
# File Path #
#############

## Gui input ##
sg.theme('DarkAmber')

# GUI layout
layout = [[sg.Text('Select the data folder + operator(s)')],
          [sg.Text('Data folder:', size=(14, 1)), sg.In(key="datadir"), sg.FolderBrowse()],
          [sg.Text('')],
          [sg.Text('Operator 1:', size=(14, 1)), sg.In(key="op1", size=(5, 1)),
           sg.Text('      Operator 2:', size=(14, 1)), sg.In(key="op2", size=(5, 1))],
          [sg.Text('Comments:', size=(14, 1)), sg.In(key="comment")],
          [sg.Text('')],
          [sg.Button('Ok'), sg.Button('Cancel')]]

# Create the Window
window = sg.Window('kV Field Size Analysis', layout)

while True:
    event, values = window.read()

    if event == 'Cancel':
        sys.exit()

    # Data folder path
    path = list(values.items())[0][1]
    data_path = path.replace('/', '\\\\')

    # Operators
    op1 = list(values.items())[-3][1]
    op2 = list(values.items())[-2][1]
    operator = [op1, op2]

    # Comments
    comment = list(values.items())[-1][1]

    if event == 'Ok':
        break

window.close()

## File path ##
measurement = data_path + "\\*.bmp"
graph = data_path + "\\kV Field Size({}).png"

########################################################################################################################

#############
# Tolerance #
#############
# Values from XRV-4000 active script
p2mm = 1 / ((3.6090 + 3.6140) / 2)
laser_x = 799.6364
laser_y = 601.8524

# kV panel size at isocentre
AB = 2700 * np.tan(np.arctan(200 / 3700))  # unit:mm
GT = 2700 * np.tan(np.arctan(150 / 3700))

# Panel size at iso in px
PA = round(laser_x - AB/p2mm) # unit:px
PB = round(AB/p2mm + laser_x)
PG = round(laser_y - GT/p2mm)
PT = round(GT/p2mm + laser_y)

# Tolerance
limit = 27 # IPEM report 91 suggested limit at 2.7 m source to detector distance
VA = round(laser_x - (AB + limit) / p2mm)  # unit:px
VB = round((AB + limit) / p2mm + laser_x)
VG = round(laser_y - (GT + limit) / p2mm)
VT = round((GT + limit) / p2mm + laser_y)

########################################################################################################################

####################
# Threshold Method #
####################
def find_v_edge(image):
    """
    This function will take an an 8 bit grayscale image and output a
    threshold image. It will also find two pairs of arrays that define the front
    and rear vertical edges of the kV field.

    input:
        image:         input image, where image must be an 8 bit grayscale image

    output:
        thresh_img:    array for the binary threshold image created by Otsu's method

        front_v_edge:  array of the horizontal pixel index position of the front
                       vertical edge

        rear_v_edge:   array of the horizontal pixel index position of the rear
                       vertical edge

        front_pos:     array of the vertical pixel index position of the front
                       vertical edge

        rear_pos:      array of the vertical pixel index position of the rear
                       vertical edge
    """

    ###############
    # Otsu Method #
    ###############
    # Normalise image
    image = image / image.max()
    image = image * 255
    image = np.round(image)

    # Create binary image using Otsu's method
    ret, thresh_img = cv2.threshold(image.astype('uint8'), 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ##################
    # Edge detection #
    ##################
    front_v_edge = []  # empty array for the vertical front edge (horizontal position)
    rear_v_edge = []  # empty array for the vertical rear edge (horizontal position)

    # Finding all points on the vertical edges of the threshold image
    for i in range(thresh_img.shape[0]):  # going through the 1200 row
        cal = []
        if np.average(thresh_img[i]) == 0:  # returning zeroes if no bright pixels could be found
            front_v_edge.append(0)
            rear_v_edge.append(0)
        else:
            cal = np.nonzero(thresh_img[i])  # finding all bright pixels within the row
            front_v_edge.append(cal[0][0])  # appending the first bright pixel
            rear_v_edge.append(cal[0][len(cal[0]) - 1])  # appending the last bright pixel

    # finding the upper bound of the front vertical edge
    for i in range(len(front_v_edge)):
        if front_v_edge[i] == 0:
            continue
        else:
            upper_front = i
            break
    # finding the lower bound of the front vertical edge
    for i in reversed(range(len(front_v_edge))):
        if rear_v_edge[i] == 0:
            continue
        else:
            lower_front = i + 1
            break
    # finding the upper bound of the rear vertical edge
    for i in range(len(rear_v_edge)):
        if rear_v_edge[i] == 0:
            continue
        else:
            upper_rear = i
            break
    # finding the lower bound of the rear vertical edge
    for i in reversed(range(len(rear_v_edge))):
        if rear_v_edge[i] == 0:
            continue
        else:
            lower_rear = i + 1
            # Final arrays of the front and rear edge
            front_v_edge = front_v_edge[upper_front:lower_front]
            rear_v_edge = rear_v_edge[upper_rear:lower_rear]

            # Arrays of the position of the edges
            front_pos = np.arange(upper_front, lower_front)
            rear_pos = np.arange(upper_rear, lower_rear)
            break

    return thresh_img, front_v_edge, rear_v_edge, front_pos, rear_pos

########################################################################################################################

###################
# Gradient Method #
###################
def gradient_images(blur_img, kernel_size):
    """
    This function will take a grayscale image and output arrays of the pixel
    intensity first and second derivatives.

    input:
        blur_img:     input image, where image must be 2D numpy array

        kernel_size:  size of kernel used to calculate image gradient


    output:
        dx:         horizontal gradient array

        dy:         vertical gradient array

        grad_img:   magnitude gradient array

        ddx:        horizontal second derivative array (gradient of gradient)

        ddy:        vertical second derivative arrayd

        dgrad_img:  magnitude of second derivative array
    """

    step = kernel_size // 2
    # array subtraction for first differential
    dy = np.zeros(blur_img.shape)
    dy[:, step:-step, :] = blur_img[:, kernel_size:, :] - blur_img[:, :-kernel_size, :]
    dx = np.zeros(blur_img.shape)
    dx[:, :, step:-step] = blur_img[:, :, kernel_size:] - blur_img[:, :, :-kernel_size]
    grad_img = np.sqrt(dx ** 2 + dy ** 2)  # calculate magnitude of gradients

    # array subtraction for second differential
    ddy = np.zeros(blur_img.shape)
    ddy[:, step:-step, :] = dy[:, kernel_size:, :] - dy[:, :-kernel_size, :]
    ddx = np.zeros(blur_img.shape)
    ddx[:, :, step:-step] = dx[:, :, kernel_size:] - dx[:, :, :-kernel_size]
    dgrad_img = np.sqrt(ddx ** 2 + ddy ** 2)  # calculate magnitude of gradients

    return dx, dy, grad_img, ddx, ddy, dgrad_img

########################################################################################################################

########
# Data #
########
path = measurement

img = [] # raw image array
name = [] # labels for the image

# find all files and load them into img using a for loop
for fname in glob.glob(path):
    # labeling files
    name.append(os.path.basename(fname)[:-4])
    # reading files
    img.append(cv2.imread(fname, cv2.IMREAD_GRAYSCALE))
# convert the list into an array
img = np.array(img, dtype='float32')

# Flipping all array A images
for img_no in np.arange(len(img)):
    flip = name[img_no]
    if flip.find("A") != -1:
        img[img_no] = np.flip(img[img_no])
    else:
        continue

# Applying Gaussian blur to all image
blur_img = [] # create an empty list`
for i in range(img.shape[0]): # lopo through images
    blur_img.append(cv2.GaussianBlur(img[i, :, :], (21, 21), 0)) # blur with a 21x21 pixel kernel
blur_img = np.array(blur_img)

########################################################################################################################

####################
# Threshold Method #
####################

# pixel indices
A_pxField = []
B_pxField = []
G_pxField = []
T_pxField = []

for img_no in np.arange(len(blur_img)):

    result, x_front, x_rear, y_front, y_rear = find_v_edge(blur_img[img_no])

    ##############################################################################
    # Slicing the vertical edge array to avoid overstreching, indices were found using visual inspection of data
    x_front = x_front[20:-20]
    y_front = y_front[20:-20]
    x_rear = x_rear[20:-20]
    y_rear = y_rear[20:-20]
    ##############################################################################
    # Finding the horizontal edges
    front_h_edge = []
    rear_h_edge = []

    # Finding all points on the horizontal edges of the thershold image
    result = result.transpose()
    for i in range(result.shape[0]):  # going through the 1600 columns
        cal = []
        if np.average(result[i]) == 0:  # returning zeroes if no bright pixels could be found
            front_h_edge.append(0)
            rear_h_edge.append(0)
        else:
            cal = np.nonzero(result[i])  # finding all bright pixels within the row
            front_h_edge.append(cal[0][0])  # appending the first bright pixel
            rear_h_edge.append(cal[0][len(cal[0]) - 1])  # appending the last bright pixel
    ##############################################################################
    # Defining the bottom edge using the vertical edges
    x_bottom = np.arange(x_front[0] + 1, x_rear[0])
    y_bottom = front_h_edge[x_bottom[0] - 1:x_bottom[-1]]

    # Defining the top edge using the vertical edges
    x_top = np.arange(x_front[-1] + 1, x_rear[-1])
    y_top = rear_h_edge[x_top[0] - 1:x_top[-1]]
    ##############################################################################
    # Binning the edges
    A_pxField.append(round(np.average(x_front)))
    B_pxField.append(round(np.average(x_rear)))
    G_pxField.append(round(np.average(y_bottom)))
    T_pxField.append(round(np.average(y_top)))

# mm size
A_mmField = []
B_mmField = []
G_mmField = []
T_mmField = []

for i in np.arange(len(A_pxField)):
    A_mmField.append(round((laser_x - A_pxField[i]) * p2mm))
    B_mmField.append(round((B_pxField[i] - laser_x) * p2mm))
    G_mmField.append(round((laser_y - G_pxField[i]) * p2mm))
    T_mmField.append(round((T_pxField[i] - laser_y) * p2mm))

########################################################################################################################

###################
# Gradient Method #
###################
# field edge location datasets
a_pk = []
b_pk = []
g_pk = []
t_pk = []

# Loop through all kV images
kernel_size = 40
for i in range(blur_img.shape[0]):
    # calculate 2nd derivative gradient images
    _, _, _, ddx, ddy, _ = gradient_images(blur_img, kernel_size)

    # manually subtract unwanted image features, i.e. BBs and noisy border
    # null BB rows & cols
    ddx[:, 550:670, :] = 0  # np.nan # horizontal null
    ddx[:, :, 750:870] = 0  # np.nan # vertical null
    ddy[:, 550:670, :] = 0  # np.nan # horizontal null
    ddy[:, :, 750:870] = 0  # np.nan # vertical null
    # null FoV rows & cols
    ddx[:, :, :50] = 0  # np.nan # horizontal null
    ddx[:, :, -51:] = 0  # np.nan # vertical null
    ddx[:, :50, :] = 0  # np.nan # horizontal null
    ddx[:, -51:, :] = 0  # np.nan # vertical null
    ddy[:, :, :50] = 0  # np.nan # horizontal null
    ddy[:, :, -51:] = 0  # np.nan # vertical null
    ddy[:, :50, :] = 0  # np.nan # horizontal null
    ddy[:, -51:, :] = 0  # np.nan # vertical null

    # locate the minima in each quadrant
    x_profile = np.median(ddx[i, ...], axis=0)
    y_profile = np.median(ddy[i, ...], axis=-1)
    a_pk.append(round(np.argmin(x_profile[:801])))
    b_pk.append(round(np.argmin(x_profile[800:]) + 800))
    g_pk.append(round(np.argmin(y_profile[:601])))
    t_pk.append(round(np.argmin(y_profile[600:]) + 600))

# mm size
a_mmpk = []
b_mmpk = []
g_mmpk = []
t_mmpk = []

for i in np.arange(len(a_pk)):
    a_mmpk.append(round((laser_x - a_pk[i]) * p2mm))
    b_mmpk.append(round((b_pk[i] - laser_x) * p2mm))
    g_mmpk.append(round((laser_y - g_pk[i]) * p2mm))
    t_mmpk.append(round((t_pk[i] - laser_y) * p2mm))

########################################################################################################################

################
# Result (CSV) #
################

table = data_path + "\\{}_{}_Result Table.csv".format(name[0][7:9], name[0][0:6]) # table file name

# Result table layout
if np.size(A_mmField) == 1:
    data = {"Side": ["A", "B", "G", "T"],
            "Threshold ({})".format(name[0][-1]): [A_mmField[0], B_mmField[0], G_mmField[0], T_mmField[0]],
            "Gradient ({})".format(name[0][-1]): [a_mmpk[0], b_mmpk[0], g_mmpk[0], t_mmpk[0]],
            }

else:
    data = {"Side": ["A", "B", "G", "T"],
            "Threshold ({})".format(name[0][-1]): [A_mmField[0], B_mmField[0], G_mmField[0], T_mmField[0]],
            "Threshold ({})".format(name[1][-1]): [A_mmField[1], B_mmField[1], G_mmField[1], T_mmField[1]],
            "Gradient ({})".format(name[0][-1]): [a_mmpk[0], b_mmpk[0], g_mmpk[0], t_mmpk[0]],
            "Gradient ({})".format(name[1][-1]): [a_mmpk[1], b_mmpk[1], g_mmpk[1], t_mmpk[1]]
            }

# Save table as CSV
df = pd.DataFrame(data)
df.to_csv(table, index=False, header=True)

################
# Result Graph #
################

i = 0
while i < np.size(A_pxField):
    ## FOV ##
    # AB
    PAB_y = np.arange(1200)[PG:PT + 1]
    PA_x = np.ones(np.size(PAB_y)) * PA
    PB_x = np.ones(np.size(PAB_y)) * PB

    # GT
    PGT_x = np.arange(1600)[PA:PB + 1]
    PG_y = np.ones(np.size(PGT_x)) * PG
    PT_y = np.ones(np.size(PGT_x)) * PT

    ## Tolerance ##
    # AB
    VAB_y = np.arange(1200)[VG:VT + 1]
    VA_x = np.ones(np.size(VAB_y)) * VA
    VB_x = np.ones(np.size(VAB_y)) * VB

    # GT
    VGT_x = np.arange(1600)[VA:VB + 1]
    VG_y = np.ones(np.size(VGT_x)) * VG
    VT_y = np.ones(np.size(VGT_x)) * VT

    ## Threshold ##
    # AB
    AB_thre_y = np.arange(1200)[G_pxField[i]:T_pxField[i] + 1]
    A_thre_x = np.ones(np.size(AB_thre_y)) * A_pxField[i]
    B_thre_x = np.ones(np.size(AB_thre_y)) * B_pxField[i]

    # GT
    GT_thre_x = np.arange(1600)[A_pxField[i]:B_pxField[i] + 1]
    G_thre_y = np.ones(np.size(GT_thre_x)) * G_pxField[i]
    T_thre_y = np.ones(np.size(GT_thre_x)) * T_pxField[i]

    ## Gradient ##
    # AB
    AB_grad_y = np.arange(1200)[g_pk[i]:t_pk[i] + 1]
    A_grad_x = np.ones(np.size(AB_grad_y)) * a_pk[i]
    B_grad_x = np.ones(np.size(AB_grad_y)) * b_pk[i]

    # GT
    GT_grad_x = np.arange(1600)[a_pk[i]:b_pk[i] + 1]
    G_grad_y = np.ones(np.size(GT_grad_x)) * g_pk[i]
    T_grad_y = np.ones(np.size(GT_grad_x)) * t_pk[i]

    ## plot ##
    plt.figure(figsize=(10, 5))
    # kV image
    plt.imshow(blur_img[i, :, :], cmap='gray')

    # Panel size at isocentre
    plt.plot(PA_x, PAB_y, color='r', linestyle='-.', label='Field of View')
    plt.plot(PB_x, PAB_y, color='r', linestyle='-.')
    plt.plot(PGT_x, PG_y, color='r', linestyle='-.')
    plt.plot(PGT_x, PT_y, color='r', linestyle='-.')

    # Tolerance
    plt.plot(VA_x, VAB_y, color='r', linestyle=':', label='Limit')
    plt.plot(VB_x, VAB_y, color='r', linestyle=':')
    plt.plot(VGT_x, VG_y, color='r', linestyle=':')
    plt.plot(VGT_x, VT_y, color='r', linestyle=':')

    # threshold
    plt.plot(A_thre_x, AB_thre_y, color='b', linestyle='--', label='Threshold Method')
    plt.plot(B_thre_x, AB_thre_y, color='b', linestyle='--')
    plt.plot(GT_thre_x, G_thre_y, color='b', linestyle='--')
    plt.plot(GT_thre_x, T_thre_y, color='b', linestyle='--')

    # gradient
    plt.plot(A_grad_x, AB_grad_y, color='c', linestyle='--', label='Gradient Method')
    plt.plot(B_grad_x, AB_grad_y, color='c', linestyle='--')
    plt.plot(GT_grad_x, G_grad_y, color='c', linestyle='--')
    plt.plot(GT_grad_x, T_grad_y, color='c', linestyle='--')

    # graph setting
    plt.title("kV field size for Imaging System {}".format((name[i][10:11])))
    plt.xlabel("AB axis / px")
    plt.ylabel("TG axis / px")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(graph.format((name[i][10:11])))

    i = i + 1

########################################################################################################################

##########
# Report #
##########

report = data_path + "\\{}_{}_Report.pdf".format(name[0][7:9], name[0][0:6]) # report file name

pdf = FPDF()
pdf.add_page()

## Title ##
pdf.set_font('Arial', 'B', 16)
pdf.cell(30, 10, 'kV Field Size Report', 'C')
pdf.ln(15)

## Session Summary ##
pdf.set_font('Arial', 'BU', 11)
pdf.cell(30, 10, 'Session Summary', 'C') # title
pdf.ln(7)

pdf.set_font('Arial', '', 10)
pdf.cell(30, 10, 'Data acquired on: {}/{}/20{}'.format(name[0][4:6], name[0][2:4], name[0][0:2]), 'C') # date
pdf.ln(7)
pdf.cell(30, 10, 'Location: Gantry {}'.format(name[0][8:9]), 'C') # gantry
pdf.ln(7)
pdf.cell(30, 10, 'Operator(s):  {} {}'.format(operator[0], operator[1]), 'C') # operator
pdf.ln(7)
pdf.cell(30, 10, 'Comments:  {}'.format(comment), 'C') # operator
pdf.ln(12)

## Tables ##
pdf.set_font('Arial', 'BU', 11)
pdf.cell(30, 10, 'Results', 'C') # title
pdf.ln(10)

# Header
if np.size(A_mmField) == 1:
    heading = ['Side', 'Threshold ({}) / mm'.format(name[0][-1]), 'Gradient ({}) / mm'.format(name[0][-1])]
    pdf.set_font('Arial', '', 10)
else:
    heading = ['Side', 'Threshold (A) / mm', 'Threshold (B) / mm', 'Gradient (A) / mm', 'Gradient (B) / mm']
    pdf.set_font('Arial', '', 10)

for i in heading:
    if i == "Side":
        pdf.cell(12, 5, str(i), align='C', border=1) # smaller cell for the first column
    else:
        pdf.cell(33, 5, str(i), align='C', border=1)
pdf.ln(5)

# Results
result_table = df.values.tolist()

for row in result_table:
    for datum in row:
        if type(datum) is str:
            pdf.cell(12, 5, str(datum), align='C', border=1) # smaller cell for the first column
        else:
            pdf.cell(33, 5, str(datum), align='C', border=1)
    pdf.ln(5)

## Graphs ##
pdf.ln(5)
for i in np.arange(len(name)):
    pdf.image(graph.format((name[i][10:11])), x=-10, h=85)

pdf.output(report, 'F')
