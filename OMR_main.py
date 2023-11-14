import cv2
import numpy as np
import utlis


#################################
path = "7.jpg"
widthImg = 700
heightImg = 700
questoes = 20
escolhas = 5
respostas = [1,2,0,3,4,2,1,4,3,3,2,0,1,1,1,1,2,0,0,4]
#################################


img = cv2.imread(path)

# PRE PROCESSAMENTO
img = cv2.resize(img,(widthImg,heightImg))
imgContours = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
imgCanny = cv2.Canny(imgBlur,10,50)

# PROCURANDO TODOS CONTORNOS
contours, hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours,contours, -1,(0,255,0),10)
# PROCURANDO RETANGULOS
rectCon= utlis.rectContour(contours)
biggestContour = utlis.getCornerPoints(rectCon[0])
gradePoints = utlis.getCornerPoints(rectCon[1])
#print(biggestContour)

if biggestContour.size !=0 and gradePoints.size !=0:
    cv2.drawContours(imgBiggestContours,biggestContour,-1,(0,255,0),20)
    cv2.drawContours(imgBiggestContours,gradePoints,-1,(255, 0, 0),20)

    biggestContour=utlis.reorder(biggestContour)
    gradePoints = utlis.reorder(gradePoints)

    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0,0],[widthImg, 0], [0, heightImg],[widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(pt1,pt2)
    imgWarpColored= cv2.warpPerspective(img, matrix,(widthImg, heightImg))

    ptG1 = np.float32(gradePoints)
    ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
    matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
    imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))
    #cv2.imshow("Grade", imgGradeDisplay)

    #APLICAR LIMITE
    imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
    imgTresh = cv2.threshold(imgWarpGray,140,255,cv2.THRESH_BINARY_INV) [1]

    boxes = utlis.splitBoxes(imgTresh)
    #cv2.imshow("Teste", boxes[2])
    print(cv2.countNonZero(boxes[1]),cv2.countNonZero(boxes[2]))

#OBTER VALORES DE PIXEL DIFERENTE DE ZERO
    myPixelVal = np.zeros((questoes,escolhas))
    countC= 0
    countR= 0

    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC +=1
        if (countC == escolhas):countR +=1 ;countC=0
    #print(myPixelVal)

#MOSTRANDO OS VALORES DAS MARCAÇÕES
    myIndex = []
    for x in range (0, questoes):
        arr = myPixelVal[x]
        print("arr", arr)
        myIndexVal = np.where(arr==np.amax(arr))
        #print(myIndexVal[0])
        myIndex.append(myIndexVal[0][0])
    #print(myIndex)

    #NOTAS
    grading=[]
    for x in range(0,questoes):
        if respostas[x] == myIndex[x]:
            grading.append(1)
        else: grading.append(0)
    #print(grading)
    nota = (sum(grading)/questoes) *100 #NOTA FINAL
    print(nota)

    #MOSTRAR RESPOSTAS
    imgResult = imgWarpColored.copy()
    imgResult = utlis.showAnswers(imgResult, myIndex, grading, respostas, questoes, escolhas)

    imgBlank = np.zeros_like(img)
    imageArray = ([img,imgGray,imgBlur,imgCanny],
            [imgContours,imgBiggestContours,imgWarpColored,imgTresh],
            [imgResult,imgBlank,imgBlank,imgBlank])
    imgStacked = utlis.stackImages(imageArray,0.3)

cv2.imshow("Stacked Images",imgStacked)
cv2.waitKey(0)