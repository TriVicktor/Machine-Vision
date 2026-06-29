#Trần Minh Nhật - 23146022
#Trịnh Minh Trí - 23146039
import cv2 as cv
import numpy as np

#_____________________________________________________Add Video_____________________________________
cap = cv.VideoCapture("Final/Badmintan.mp4")
if not cap.isOpened():
    print('can not open video clip/camera')
    exit()

ret, frame = cap.read()
#1. Add ảnh và resize
img = frame.copy()
width = img.shape[1]
height = img.shape[0]
img = cv.resize(img, (int(width/2), int(height/2)))

#_______________________________________________Phần Detect Sân và Warp nửa sân________________________________________
#2. Tiền xử lí HSV và find contour và cắt ảnh sân xanh
HSV_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
Gaus_img = cv.GaussianBlur(HSV_img, (5, 5), 0)
lower_Ground = np.array([65, 30, 80])
upper_Ground = np.array([90, 120, 225])
b_HSV = cv.inRange(Gaus_img, lower_Ground, upper_Ground, cv.THRESH_BINARY)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
morph_img = cv.morphologyEx(b_HSV, cv.MORPH_OPEN, kernel, iterations=1)
morph_img = cv.morphologyEx(morph_img, cv.MORPH_CLOSE, kernel, iterations=2)
contours, _ = cv.findContours(morph_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv.contourArea)
x_g, y_g, w_g, h_g = cv.boundingRect(cnt)
cropped = HSV_img[y_g:y_g+h_g, x_g:x_g+w_g]
cropped_img = img[y_g:y_g+h_g, x_g:x_g+w_g]


#3. Tiền xử lí lần 2: cắt khung trắng
gaus_cropped = cv.GaussianBlur(cropped, (5,5), 0)
b_cropped = cv.inRange(gaus_cropped, (60,0,130), (90,70,255))
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
morph_cropped = cv.morphologyEx(b_cropped, cv.MORPH_CLOSE, kernel, iterations=2)
contours, _ = cv.findContours(morph_cropped,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv.contourArea)
x_g2, y_g2, w_g2, h_g2 = cv.boundingRect(cnt)
cropped_2 = b_cropped[y_g2:y_g2+h_g2, x_g2:x_g2+w_g2]
cropped_img_2 = cropped_img[y_g2:y_g2+h_g2, x_g2:x_g2+w_g2]


#4. Thuật toán Canny & Hough Line
canny = cv.Canny(cropped_2, 50, 100)
cv.imshow("canny", canny)
lines = cv.HoughLinesP(canny, rho=1, theta=np.pi/180,threshold=15, minLineLength=250,maxLineGap=30)
line_img = cropped_img_2.copy()
vertical_lines = []
count = 0
if lines is not None:
    for line in lines:
        x1,y1,x2,y2 = line[0]
        angle = abs( np.degrees( np.arctan2(y2-y1, x2-x1)))
        if 5 < abs(angle) < 95:
            vertical_lines.append(line[0])
            count += 1

    #Tìm 2 đường biên
    line_bien = []
    for line in vertical_lines:
        x1, y1, x2, y2 = line
        x_avg = (x1 + x2)/2
        line_bien.append((x_avg, line))
    line_bien.sort(key=lambda item: item[0])
    left_outer = line_bien[0][1]
    right_outer = line_bien[-1][1]
    x1_l,y1_l,x2_l,y2_l = left_outer
    x1_r,y1_r,x2_r,y2_r = right_outer

    if y1_l < y2_l:
        top_left = [x1_l, y1_l]
        bottom_left = [x2_l, y2_l]
    else:
        top_left = [x2_l, y2_l]
        bottom_left = [x1_l, y1_l]

    if y1_r < y2_r:
        top_right = [x1_r, y1_r]
        bottom_right = [x2_r, y2_r]
    else:
        top_right = [x2_r, y2_r]
        bottom_right = [x1_r, y1_r]

    mid_left = [
        (top_left[0] + bottom_left[0])//2,
        (top_left[1] + bottom_left[1])//2
    ]
    mid_right = [
        (top_right[0] + bottom_right[0])//2,
        (top_right[1] + bottom_right[1])//2
    ]
else:
    print(f"Không tìm thấy Line")


#5. Warp Transform
width_warp = 700
height_warp = 700
pts_ref = np.float32([
    mid_left,
    mid_right,
    bottom_right,
    bottom_left
])
pts_dst = np.float32([
    [0, 0],
    [width_warp-1, 0],
    [width_warp-1, height_warp-1],
    [0, height_warp-1]
])
M = cv.getPerspectiveTransform(pts_ref, pts_dst)
warped = cv.warpPerspective(cropped_img_2, M, (width_warp, height_warp))


#______________________________________Phần Detect người________________________________
def detect_player(img,
                  x_g, y_g, w_g, h_g,
                  lower_Ground,
                  upper_Ground):
    HSV_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cropped = HSV_img[y_g:y_g+h_g,
                      x_g:x_g+w_g]
    cropped_img = img[y_g:y_g+h_g,
                      x_g:x_g+w_g]
    mask_ground = cv.inRange(cropped, lower_Ground, upper_Ground)
    mask_human = cv.bitwise_not(mask_ground)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    mask_human = cv.morphologyEx(mask_human, cv.MORPH_OPEN, kernel, iterations=2)
    mask_human = cv.morphologyEx(mask_human, cv.MORPH_CLOSE, kernel,iterations=1)

    contours,_ = cv.findContours(mask_human,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    center_x = cropped_img.shape[1]//2
    center_y = cropped_img.shape[0]//2
    best_contour = None
    min_dist = float('inf')
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 2000 or area > 4500:
            continue
        x,y,w,h = cv.boundingRect(cnt)
        ratio = h / max(w,1)
        if ratio < 1.2 or ratio > 4:
            continue
        cx = x + w//2
        cy = y + h//2
        dist = np.sqrt(
            (cx-center_x)**2 +
            (cy-center_y)**2
        )
        if dist < min_dist:
            min_dist = dist
            best_contour = cnt
    if best_contour is None:
        return None

    x,y,w,h = cv.boundingRect(best_contour)
    left_pad = 10
    right_pad = 10
    top_pad = 5
    bottom_pad = 25
    x = max(0,x-left_pad)
    y = max(0,y-top_pad)
    w = min(
        cropped_img.shape[1]-x,
        w+left_pad+right_pad
    )
    h = min(
        cropped_img.shape[0]-y,
        h+top_pad+bottom_pad
    )
    x += x_g
    y += y_g
    return (x,y,w,h)


frame_count = 0
lost = False
tracker = None
if tracker is None:
    bbox = detect_player(
    img,
    x_g, y_g, w_g, h_g,
    lower_Ground,
    upper_Ground
    )
    if bbox is not None:
        tracker = cv.legacy.TrackerCSRT_create()
        tracker.init(img, bbox)
        init_area = bbox[2] * bbox[3]


def find_foot_point(img,
                    bbox,
                    lower_Ground,
                    upper_Ground):
    x, y, w, h = [int(v) for v in bbox]
    start_y = int(h * 0.65)
    foot_roi = img[
        y + start_y : y + h,
        x : x + w
    ]
    if foot_roi.size == 0:
        return None
    hsv = cv.cvtColor(foot_roi,cv.COLOR_BGR2HSV)
    mask_ground = cv.inRange(hsv,lower_Ground,upper_Ground)
    mask_player = cv.bitwise_not(mask_ground)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    mask_player = cv.morphologyEx(mask_player,cv.MORPH_OPEN,kernel,iterations=1)
    mask_player = cv.morphologyEx(mask_player,cv.MORPH_CLOSE,kernel,iterations=1)
    contours, _ = cv.findContours(
        mask_player,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return None
    cnt = max(
        contours,
        key=cv.contourArea
    )
    ys = cnt[:,:,1]
    max_y = np.max(ys)
    idx = np.where(ys >= max_y - 25)[0]
    bottom_pts = cnt[idx][:,0,:]
    if len(bottom_pts) < 5:
        return None

    bx = np.min(bottom_pts[:,0])
    by = np.min(bottom_pts[:,1])
    bw = np.max(bottom_pts[:,0]) - bx
    bh = np.max(bottom_pts[:,1]) - by

    foot_x_local = bx + bw//2
    foot_y_local = by + bh//2
    foot_x = x + foot_x_local
    foot_y = (
        y
        + start_y
        + foot_y_local
    )
    return(foot_x, foot_y, bw, bh)


#_________________________________________________________Phần Heatmap________________________________________________
# Giá trị tích lũy tối đa cho mỗi pixel. Khi 1 vị trí đạt mức này thì
# coi như đã "no" (màu đỏ đậm nhất của JET) và không cộng thêm nữa.
HEATMAP_MAX = 255.0

def init_heatmap(height, width):
    """Khởi tạo ma trận tích lũy nhiệt trống (toàn bộ = 0 -> sẽ hiện màu xanh biển)."""
    return np.zeros((height, width), dtype=np.float32)


def _ellipse_to_warp_polygon(foot_x, foot_y, bw, bh,
                              x_g, x_g2, y_g, y_g2,
                              M, num_pts=24):
    """
    Lấy các điểm trên viền ellipse chân (tâm foot_x, foot_y; bán trục
    max(15, bw/2) và max(5, bh/2)) ở hệ tọa độ ảnh img, quy đổi về hệ tọa độ cropped_img_2, rồi chiếu qua homography M.
    """
    axis_x = max(15, bw // 2)
    axis_y = max(5, bh // 2)

    angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
    pts = []
    for a in angles:
        ex = foot_x + axis_x * np.cos(a)
        ey = foot_y + axis_y * np.sin(a)
        pts.append([ex - x_g - x_g2, ey - y_g - y_g2])

    pts = np.array([pts], dtype=np.float32)          # shape (1, N, 2)
    warped_pts = cv.perspectiveTransform(pts, M)[0]    # shape (N, 2)
    return np.round(warped_pts).astype(np.int32)


def update_heatmap(heatmap_accum, foot_point, foot_size,
                    x_g, x_g2, y_g, y_g2,
                    M, width_warp, height_warp,
                    max_value=HEATMAP_MAX, ink=2.0):
    """
    Vẽ TOÀN BỘ ellipse chân vào heatmap, sau khi chiếu qua homography M. Mỗi frame cộng thêm `ink` vào vùng ellipse đó,
    nhưng clamp (np.clip) tại max_value để tránh cộng dồn vô hạn / tràn -
    khi 1 pixel đã đạt max_value thì giữ nguyên, không cộng thêm nữa.
    """
    if foot_point is None or foot_size is None:
        return heatmap_accum

    foot_x, foot_y = foot_point
    bw, bh = foot_size

    warp_poly = _ellipse_to_warp_polygon(
        foot_x, foot_y, bw, bh,
        x_g, x_g2, y_g, y_g2, M
    )

    # Bỏ qua nếu polygon nằm hoàn toàn ngoài khung warp
    x_min, y_min = warp_poly[:, 0].min(), warp_poly[:, 1].min()
    x_max, y_max = warp_poly[:, 0].max(), warp_poly[:, 1].max()
    if x_max < 0 or y_max < 0 or x_min >= width_warp or y_min >= height_warp:
        return heatmap_accum

    temp_mask = np.zeros_like(heatmap_accum)
    cv.fillPoly(temp_mask, [warp_poly], ink)
    temp_mask = cv.GaussianBlur(temp_mask, (15, 15), 0)

    # Cộng dồn nhưng chặn (clamp) ở max_value để không bị trôi/tràn
    heatmap_accum = np.clip(heatmap_accum + temp_mask, 0, max_value)

    return heatmap_accum


def generate_paper_heatmap(heatmap_accum, warped_frame, height_warp, width_warp,
                            max_value=HEATMAP_MAX, threshold=3, alpha=0.55):
    """
    Phủ heatmap LÊN ẢNH SÂN ĐÃ WARP (warped_frame)
    - Thang màu JET dùng mốc CỐ ĐỊNH 0 -> max_value (không normalize động theo max hiện tại), nên màu không bị "trôi/nhạt" theo thời gian.
    - Vùng chưa có ai đi qua (giá trị ~0) -> giữ nguyên ảnh sân gốc.
    - Vùng có tích lũy -> chồng màu JET lên ảnh sân:
        xanh biển = di chuyển ít, xanh lá = trung bình, đỏ = nhiều
        (đỏ đậm nhất khi đạt max_value, vì đã clamp ở update_heatmap).
    """
    heatmap_clamped = np.clip(heatmap_accum, 0, max_value)
    heatmap_8u = (heatmap_clamped / max_value * 255).astype(np.uint8)
    heatmap_color = cv.applyColorMap(heatmap_8u, cv.COLORMAP_JET)

    # Mask vùng thực sự có dữ liệu (đã có người đi qua), bỏ threshold để loại nhiễu nhỏ
    _, motion_mask = cv.threshold(heatmap_8u, threshold, 255, cv.THRESH_BINARY)
    motion_mask_inv = cv.bitwise_not(motion_mask)

    # Vùng không có dữ liệu: giữ nguyên ảnh sân
    bg = cv.bitwise_and(warped_frame, warped_frame, mask=motion_mask_inv)

    # Vùng có dữ liệu: chồng heatmap màu lên ảnh sân theo alpha
    blended = cv.addWeighted(warped_frame, 1 - alpha, heatmap_color, alpha, 0)
    fg = cv.bitwise_and(blended, blended, mask=motion_mask)

    return cv.add(bg, fg)


def generate_heatmap_with_colorbar(heatmap_accum, warped_frame, height_warp, width_warp,
                                    max_value=HEATMAP_MAX):
    """Ảnh heatmap cuối cùng (chồng lên ảnh sân) kèm thanh thang màu."""
    heatmap_img = generate_paper_heatmap(heatmap_accum, warped_frame, height_warp, width_warp, max_value)

    colorbar_width = 70
    colorbar_scale = np.linspace(255, 0, height_warp).astype(np.uint8).reshape(height_warp, 1)
    colorbar_img = cv.applyColorMap(colorbar_scale, cv.COLORMAP_JET)
    colorbar_final = cv.resize(colorbar_img, (colorbar_width, height_warp))

    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(colorbar_final, "High", (10, 30), font, 0.6, (255,255,255), 1)
    cv.putText(colorbar_final, "Low", (10, height_warp - 20), font, 0.6, (255,255,255), 1)

    margin = 30
    final_width = width_warp + margin + colorbar_width
    result_img = np.ones((height_warp, final_width, 3), dtype=np.uint8) * 255
    result_img[0:height_warp, 0:width_warp] = heatmap_img
    start_x = width_warp + margin
    result_img[0:height_warp, start_x:start_x + colorbar_width] = colorbar_final

    return result_img


heatmap_accum = init_heatmap(height_warp, width_warp)


#________________________________________________Vòng lặp video cho đến khi kết thúc_____________________________________________
while True:
    ret, frame = cap.read()
    if not ret:
        print(' can not read video frame. Video ended?')
        break

    #**********************Phần Detect và Warp nửa sân********************
    img = frame.copy()
    img = cv.resize(img, (int(width/2), int(height/2)))
    cropped_img = img[y_g:y_g+h_g, x_g:x_g+w_g]
    cropped_img_2 = cropped_img[y_g2:y_g2+h_g2, x_g2:x_g2+w_g2]
    warped = cv.warpPerspective(cropped_img_2, M, (width_warp, height_warp))

    #***********************Phần Detect lông thủ***********************
    if tracker is not None:
        success, human = tracker.update(img)
    else:
        success = False

    if not success:
        lost = True

    if success:
        x,y,w,h = [int(v) for v in human]
        cx = x + w//2
        cy = y + h//2
        if cx < x_g: lost = True
        if cx > x_g+w_g: lost = True
        if cy < y_g: lost = True
        if cy > y_g+h_g: lost = True

    if success:
        x,y,w,h = [int(v) for v in human]
        area_now = w*h
        if area_now > init_area*2: lost = True
        if area_now < init_area*0.3: lost = True

    if lost:
        tracker = None
        if tracker is None:
            bbox = detect_player(
            img,
            x_g, y_g, w_g, h_g,
            lower_Ground,
            upper_Ground
            )
            if bbox is not None:
                tracker = cv.legacy.TrackerCSRT_create()
                tracker.init(img,bbox)
                init_area = bbox[2]*bbox[3]
            lost = False

    frame_count += 1
    if frame_count % 15 == 0:
        bbox = detect_player(
            img,
            x_g, y_g, w_g, h_g,
            lower_Ground,
            upper_Ground
        )
        if bbox is not None:
            tracker = cv.legacy.TrackerCSRT_create()
            tracker.init(img,bbox)
        frame_count = 0

    #************************Phần Detect foot + cộng dồn heatmap******************************************
    foot = None
    if success:
        foot = find_foot_point(
            img,
            human,
            lower_Ground,
            upper_Ground
        )

        if foot is not None:
            foot_x, foot_y, bw, bh = foot
            cv.ellipse(img, (foot_x, foot_y),(max(15, bw//2), max(5, bh//2)), 0, 0, 360, (0,255,0), 2)

            # Cộng dồn TOÀN BỘ ellipse chân vào heatmap (tọa độ warp),
            # clamp ở HEATMAP_MAX để tránh trôi/tràn dữ liệu
            heatmap_accum = update_heatmap(
                heatmap_accum,
                (foot_x, foot_y),
                (bw, bh),
                x_g, x_g2, y_g, y_g2,
                M, width_warp, height_warp,
                max_value=HEATMAP_MAX, ink=2.0
            )

    if ret and success:
        p1 = (int(human[0]), int(human[1]))
        p2 = (int(human[0]+human[2]), int(human[1]+human[3]))
        cv.rectangle(img, p1, p2, (0,0,255), 2)

    # Heatmap realtime để theo dõi trong lúc chạy
    realtime_heatmap = generate_paper_heatmap(heatmap_accum, warped, height_warp, width_warp)

    cv.imshow('video', img)
    cv.imshow('Warped', warped)
    cv.imshow('Heatmap Realtime', realtime_heatmap)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


# Đóng các cửa sổ video / tracking / realtime, chỉ giữ lại ảnh heatmap final
cap.release()
cv.destroyWindow('video')
cv.destroyWindow('Warped')
cv.destroyWindow('Heatmap Realtime')
cv.destroyWindow('canny')

final_heatmap = generate_heatmap_with_colorbar(heatmap_accum, warped, height_warp, width_warp)
cv.imwrite('badminton_heatmap_final.png', final_heatmap)
cv.imshow('Final Heatmap', final_heatmap)

cv.waitKey(0)
cv.destroyAllWindows()