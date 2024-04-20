from mpmath import matrix

def element_pow(mat1, mat2):
    result = matrix(mat1.rows, mat1.cols)

    if type(mat2) != type(mat1):
        matTemp = result
        for i in range(mat1.rows):
            for j in range(mat1.cols):
                matTemp[i, j] = mat2
    else:
        matTemp = mat2

    for i in range(mat1.rows):
        for j in range(mat1.cols):
            result[i, j] = mat1[i, j] ** matTemp[i, j]

    return result

def element_mul(mat1, mat2):
    result = matrix(mat1.rows, mat1.cols)

    for i in range(mat1.rows):
        for j in range(mat1.cols):
            result[i, j] = mat1[i, j] * mat2[i, j]

    return result

def element_div(mat1, mat2):
    result = matrix(mat1.rows, mat1.cols)

    for i in range(mat1.rows):
        for j in range(mat1.cols):
            result[i, j] = mat1[i, j] / mat2[i, j]

    return result

