import pyautogui
import catch
def sccshot(x,y,w,h):
    img=pyautogui.screenshot(region=(x,y,w,h))
    return img

if __name__ == "__main__":
    # 左上角座標(x,y)和截圖寬高(w,h)
    x=60
    y=60
    w=1820
    h=840
    catch.catch_view(x,y,w,h)
    #scrcpy --window-x 60 --window-y 60 --window-width 1820 --window-height 840