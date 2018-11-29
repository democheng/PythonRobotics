import numpy
# reference: https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
def line_low(x0, y0, x1, y1):
    res = []
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy
    D = 2*dy - dx
    y = y0
    for x in range(x0, x1 + 1):
        res.append((x,y))
        if D > 0:
            y = y + yi
            D = D - 2*dx
        D = D + 2*dy
    return res

def line_high(x0, y0, x1, y1):
    res = []
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    D = 2*dx - dy
    x = x0

    for y in range(y0, y1 + 1):
        res.append((x,y))
        if D > 0:
            x = x + xi
            D = D - 2*dy
        D = D + 2*dx
    return res

def bresenham_line(A, B):
    x0 = int(A[0])
    y0 = int(A[1])
    x1 = int(B[0])
    y1 = int(B[1])

    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            return line_low(x1, y1, x0, y0)
        else:
            return line_low(x0, y0, x1, y1)
    else:
        if y0 > y1:
            return line_high(x1, y1, x0, y0)
        else:
            return line_high(x0, y0, x1, y1)

# reference: https://en.wikipedia.org/wiki/Midpoint_circle_algorithm
def bresenham_circle(C):
    x0 = int(C[0])
    y0 = int(C[1])
    radius = int(C[2])
    res = []
    x = int(radius-1)
    y = int(0)
    dx = int(1)
    dy = int(1)
    err = int(dx - (radius << 1))

    while (x >= y):
        res.append((x0 + x, y0 + y))
        res.append((x0 + y, y0 + x))
        res.append((x0 - y, y0 + x))
        res.append((x0 - x, y0 + y))
        res.append((x0 - x, y0 - y))
        res.append((x0 - y, y0 - x))
        res.append((x0 + y, y0 - x))
        res.append((x0 + x, y0 - y))

        if (err <= 0):
            y += 1
            err += dy
            dy += 2
        
        if (err > 0):
            x -= 1
            dx += 2
            err += dx - (radius << 1)
    
    return res
