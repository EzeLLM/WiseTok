# brake
# acceleration, seconds
# max_speed
# bhp
# deccelleration
class Car:

    def __init__(self, max_speed):
        self.speed = 0
        self.max_speed = max_speed
        self.car_started = "off"

    def car_start(self, on_or_off):
        self.car_started = on_or_off
        if self.car_started == "off":
            return False
        elif self.car_started == "on":
            return True
        else:
            return "car must be on or off"

    def accelerate(self, speed_increase, seconds):
        if self.car_start(self.car_started) is True:
            if self.speed + speed_increase > self.max_speed:
                speed_increase = self.max_speed - self.speed
                self.speed += speed_increase
                return f"You accelerate at {speed_increase/seconds} mph per second"
            else:
                self.speed += speed_increase
                return f"You accelerate at {speed_increase/seconds} mph per second"
        else:
            return "You must start the car before you can accelerate"



lambo = Car(200)
bmw = Car(150)
bugatti = Car(250)

print(lambo.car_start("off"))
print(lambo.accelerate(50, 2))
print(lambo.speed)
print(lambo.car_start("on"))
print(lambo.accelerate(500, 2))
print(lambo.speed)
