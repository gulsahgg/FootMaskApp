import cv2 as cv
import numpy as np
import itertools
import matplotlib.pyplot as plt
import imutils
import math
from math import sqrt

def generate_nonadjacent_combination(input_list,take_n):
        """
        It generates combinations of m taken n at a time where there is no adjacent n.
        INPUT:
            input_list = (iterable) List of elements you want to extract the combination
            take_n =     (integer) Number of elements that you are going to take at a time in
                         each combination
        OUTPUT:
            all_comb =   (np.array) with all the combinations
        """
        all_comb = []
        for comb in itertools.combinations(input_list, take_n):
            comb = np.array(comb)
            d = np.diff(comb)
            fd = np.diff(np.flip(comb))
            if len(d[d==1]) == 0 and comb[-1] - comb[0] != 7:
                all_comb.append(comb)
                print(comb)
        return all_comb

def populate_intersection_kernel(combinations):
        """
        Maps the numbers from 0-7 into the 8 pixels surrounding the center pixel in
        a 9 x 9 matrix clockwisely i.e. up_pixel = 0, right_pixel = 2, etc. And
        generates a kernel that represents a line intersection, where the center
        pixel is occupied and 3 or 4 pixels of the border are ocuppied too.
        INPUT:
            combinations = (np.array) matrix where every row is a vector of combinations
        OUTPUT:
            kernels =      (List) list of 9 x 9 kernels/masks. each element is a mask.
        """
        n = len(combinations[0])
        template = np.array((
                [-1, -1, -1],
                [-1, 1, -1],
                [-1, -1, -1]), dtype="int")
        match = [(0,1),(0,2),(1,2),(2,2),(2,1),(2,0),(1,0),(0,0)]
        kernels = []
        for n in combinations:
            tmp = np.copy(template)
            for m in n:
                tmp[match[m][0],match[m][1]] = 1
            kernels.append(tmp)
        return kernels
def give_intersection_kernels():
        """
        Generates all the intersection kernels in a 9x9 matrix.
        INPUT:
            None
        OUTPUT:
            kernels =      (List) list of 9 x 9 kernels/masks. each element is a mask.
        """
        input_list = np.arange(8)
        taken_n = [4,3]
        kernels = []
        for taken in taken_n:
            comb = generate_nonadjacent_combination(input_list,taken)
            tmp_ker = populate_intersection_kernel(comb)
            kernels.extend(tmp_ker)
        return kernels
def find_line_intersection(input_image, show=0):
        """
        Applies morphologyEx with parameter HitsMiss to look for all the curve
        intersection kernels generated with give_intersection_kernels() function.
        INPUT:
            input_image =  (np.array dtype=np.uint8) binarized m x n image matrix
        OUTPUT:
            output_image = (np.array dtype=np.uint8) image where the nonzero pixels
                           are the line intersection.
        """
        kernel = np.array(give_intersection_kernels())
        output_image = np.zeros(input_image.shape)
        for i in np.arange(len(kernel)):
            out = cv.morphologyEx(input_image, cv.MORPH_HITMISS, kernel[i,:,:])
            output_image = output_image + out
        if show == 1:
            show_image = np.reshape(np.repeat(input_image, 3, axis=1),(input_image.shape[0],input_image.shape[1],3))*255
            show_image[:,:,1] = show_image[:,:,1] -  output_image *255
            show_image[:,:,2] = show_image[:,:,2] -  output_image *255
            plt.imshow(show_image)
        return output_image
def find_endoflines(input_image, show=0):
        kernel_0 = np.array((
                [-1, -1, -1],
                [-1, 1, -1],
                [-1, 1, -1]), dtype="int")

        kernel_1 = np.array((
                [-1, -1, -1],
                [-1, 1, -1],
                [1,-1, -1]), dtype="int")

        kernel_2 = np.array((
                [-1, -1, -1],
                [1, 1, -1],
                [-1,-1, -1]), dtype="int")

        kernel_3 = np.array((
                [1, -1, -1],
                [-1, 1, -1],
                [-1,-1, -1]), dtype="int")

        kernel_4 = np.array((
                [-1, 1, -1],
                [-1, 1, -1],
                [-1,-1, -1]), dtype="int")

        kernel_5 = np.array((
                [-1, -1, 1],
                [-1, 1, -1],
                [-1,-1, -1]), dtype="int")

        kernel_6 = np.array((
                [-1, -1, -1],
                [-1, 1, 1],
                [-1,-1, -1]), dtype="int")

        kernel_7 = np.array((
                [-1, -1, -1],
                [-1, 1, -1],
                [-1,-1, 1]), dtype="int")
        kernel = np.array((kernel_0,kernel_1,kernel_2,kernel_3,kernel_4,kernel_5,kernel_6, kernel_7))
        output_image = np.zeros(input_image.shape)
        for i in np.arange(8):
            out = cv.morphologyEx(input_image, cv.MORPH_HITMISS, kernel[i,:,:])
            output_image = output_image + out
        if show == 1:
            show_image = np.reshape(np.repeat(input_image, 3, axis=1),(input_image.shape[0],input_image.shape[1],3))*255
            show_image[:,:,1] = show_image[:,:,1] -  output_image *255
            show_image[:,:,2] = show_image[:,:,2] -  output_image *255
            plt.imshow(show_image)
        return output_image#, np.where(output_image == 1)

def findDist(p1,p2):
        return sqrt( (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 )

def NearPoint(arr1,arr2,bot):
        minimum=1024
        point=0
        a=0
        for i in range(arr1.size):
            p=(arr1[i],arr2[i])
            a=findDist(p,bot)
            if (minimum>a):
                minimum=a
                point=p
        return point

def findPoints(img):
        cnts = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        return extLeft,extRight,extTop,extBot
