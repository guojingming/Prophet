import turtle
import time
import random

win = turtle.Screen()
win.title("I love wenxue")
win.bgcolor("orange")
win.setup(width=600, height=600)
win.tracer(0)

snake = turtle.Turtle()
snake.shape("square")
snake.color("black")
snake.penup()
snake.speed(0)
snake.goto(0, 0)  # 初始位置
snake.direction = "stop"

apple = turtle.Turtle()
apple.shape("circle")
apple.color("red")
apple.penup()
apple.speed(0)
apple.goto(100, 100)

score = 0
high_score = 0
pen = turtle.Turtle()
pen.speed(0)
pen.color("white")
pen.penup()
pen.hideturtle()
pen.goto(0, 260)
pen.write("Current score: {}   Highest score: {}".format(score, high_score), align="center", font=("Courier", 16, "normal"))

def snake_move():
    if snake.direction == "up":
        y = snake.ycor()
        snake.sety(y + 20)
    if snake.direction == "down":
        y = snake.ycor()
        snake.sety(y - 20)
    if snake.direction == "left":
        x = snake.xcor()
        snake.setx(x - 20)
    if snake.direction == "right":
        x = snake.xcor()
        snake.setx(x + 20)

def go_up():
    if snake.direction != "down":
        snake.direction = "up"
def go_down():
    if snake.direction != "up":
        snake.direction = "down"
def go_left():
    if snake.direction != "right":
        snake.direction = "left"
def go_right():
    if snake.direction != "left":
        snake.direction = "right"

win.listen()
win.onkeypress(go_up, "w")
win.onkeypress(go_down, "s")
win.onkeypress(go_left, "a")
win.onkeypress(go_right, "d")

while True:
    win.update()
    if snake.distance(apple) < 20:
        rand_num1 = random.randint(2, 14)
        rand_num2 = random.randint(2, 14)
        x = rand_num1 * 20
        y = rand_num2 * 20
        print("x:", x)
        print("y:", y)
        apple.goto(x, y)
        score += 10
        if score > high_score:
            high_score = score
        pen.clear()
        pen.write("Current score: {}  Highest score: {}".format(score, high_score), align="center", font=("Courier", 16, "normal"))
    snake_move()
    if snake.xcor() > 290 or snake.xcor() < -290 or snake.ycor() > 290 or snake.ycor() < -290:
        time.sleep(1)
        snake.goto(0, 0)
        snake.direction = "stop"
        score = 0
        pen.clear()
        pen.write("Current score: {}  Highest score: {}".format(score, high_score), align="center", font=("Courier", 16, "normal"))
    time.sleep(0.1)
win.mainloop()