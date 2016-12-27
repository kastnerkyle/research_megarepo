# Further reduced raycast example
# Based on https://github.com/Mekire/pygame-raycasting-experiment
# Original post in JS
# http://www.playfuljs.com/a-first-person-engine-in-265-lines/
# Author: Kyle Kastner
# License: BSD 3-Clause
# Instructions:
# Walk around with keyboard forward, back, left, and right keys
# Build new walls with left shift or right shift
# Destroy walls you are touching with space
import os
import sys
import math
import itertools
import random
import pygame as pg
import numpy as np
from collections import namedtuple


noise_random_state = np.random.RandomState(1337)
def perlin(filename, gradient=False):
    # Perlin Noise Generator
    # http://en.wikipedia.org/wiki/Perlin_noise
    # http://en.wikipedia.org/wiki/Bilinear_interpolation
    # FB36 - 20130222
    from PIL import Image
    imgx = 800; imgy = 600 # image size
    image = Image.new("RGB", (imgx, imgy))
    pixels = image.load()
    octaves = int(math.log(max(imgx, imgy), 2.0))
    persistence = noise_random_state.rand()
    imgAr = [[0.0 for i in range(imgx)] for j in range(imgy)] # image array
    totAmp = 0.0
    for k in range(octaves):
        freq = 2 ** k
        amp = persistence ** k
        totAmp += amp
        # create an image from n by m grid of random numbers (w/ amplitude)
        # using Bilinear Interpolation
        n = freq + 1; m = freq + 1 # grid size
        ar = [[noise_random_state.rand() * amp for i in range(n)] for j in range(m)]
        nx = imgx / (n - 1.0); ny = imgy / (m - 1.0)
        for ky in range(imgy):
            for kx in range(imgx):
                i = int(kx / nx); j = int(ky / ny)
                dx0 = kx - i * nx; dx1 = nx - dx0
                dy0 = ky - j * ny; dy1 = ny - dy0
                z = ar[j][i] * dx1 * dy1
                z += ar[j][i + 1] * dx0 * dy1
                z += ar[j + 1][i] * dx1 * dy0
                z += ar[j + 1][i + 1] * dx0 * dy0
                z /= nx * ny
                imgAr[ky][kx] += z # add image layers together
            else:
                pixels[kx, ky] = (0, 0, 255)

    # paint image
    for ky in range(imgy):
        for kx in range(imgx):
            c = int(imgAr[ky][kx] / totAmp * 255)
            scaley = int(ky / float(imgy) * 255)
            scalex = int(kx / float(imgx) * 255)
            if gradient:
                if ky > (0.5 * imgy):
                    # grass
                    r = 0
                    g = 200 - int(scaley / 1.45)
                    b = 0
                else:
                    # horizon sky
                    r = scaley
                    g = scaley
                    b = 200 - scaley
                pixels[kx, ky] = r, g, b
            else:
                r, g, b = (c, c, c)
                pixels[kx, ky] = r, g, b
    image.save(filename)

RayInfo = namedtuple("RayInfo", ["sin", "cos"])
WallInfo = namedtuple("WallInfo", ["top", "height"])
SCREEN_SIZE = (640, 480)
FIELD_OF_VIEW = 0.4 * math.pi


class Player(object):
    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.direction = direction
        self.speed = 3
        self.rotate_speed = np.pi

    def rotate(self, angle):
        self.direction = (self.direction + angle + 2 * math.pi) % (2 * math.pi)

    def walk(self, distance, game_map):
        dx = np.cos(self.direction) * distance
        dy = np.sin(self.direction) * distance
        if game_map.get(self.x + dx, self.y) <= 0:
            self.x += dx
        if game_map.get(self.x, self.y + dy) <= 0:
            self.y += dy

    def action(self, distance, game_map, action):
        dx = np.cos(self.direction) * distance
        dy = np.sin(self.direction) * distance
        game_map.set(self.x + dx, self.y + dy, action)

    def update(self, keys, dt, game_map):
        if keys[pg.K_LEFT]:
            self.rotate(-self.rotate_speed * dt)
        if keys[pg.K_RIGHT]:
            self.rotate(self.rotate_speed * dt)
        if keys[pg.K_UP]:
            self.walk(self.speed * dt, game_map)
        if keys[pg.K_DOWN]:
            self.walk(-self.speed * dt, game_map)
        if keys[pg.K_SPACE]:
            # 0 is destroy
            self.action(self.speed * dt, game_map, 0)
        if keys[pg.K_LSHIFT] or keys[pg.K_RSHIFT]:
            # 1 is build
            # + 2 is to not get stuck in a block when making it
            self.action(self.speed * dt + 2, game_map, 1)


class Image(object):
    # thin wrapper around images for access to height, width
    def __init__(self, image):
        self.image = image
        self.width, self.height = self.image.get_size()


class Map(object):
    def __init__(self, size):
        self.size = size
        random_state = np.random.RandomState(1999)
        self.random_state = random_state
        self.wall_grid = self.randomize(size)
        self.wall_texture = Image(pg.image.load("wall_texture.bmp").convert())
        sky_size = (int(SCREEN_SIZE[0] * (2 * np.pi / FIELD_OF_VIEW)),
                    SCREEN_SIZE[1])
        sky_image = pg.image.load("sky_texture.bmp").convert()
        sky_texture = pg.transform.smoothscale(sky_image, sky_size)
        self.sky_texture = Image(sky_texture)
        self.light = 0

    def get(self, x, y):
        point = (int(math.floor(x)), int(math.floor(y)))
        return self.wall_grid.get(point, -1)

    def set(self, x, y, value):
        point = (int(math.floor(x)), int(math.floor(y)))
        self.wall_grid[point] = value
        return self

    def randomize(self, size):
        coordinates = itertools.product(range(size), repeat=2)
        ratio = 0.3
        return {coord: self.random_state.rand() < ratio
                for coord in coordinates}

    def cast_ray(self, point, angle, cast_range):
        ray_info = RayInfo(math.sin(angle), math.cos(angle))
        origin = Point(point)
        ray = [origin]
        while origin.height <= 0 and origin.distance <= cast_range:
            dist = origin.distance
            step_x = origin.step(ray_info.sin, ray_info.cos)
            step_y = origin.step(ray_info.cos, ray_info.sin, invert=True)
            if step_x.length < step_y.length:
                next_step = step_x.inspect(ray_info, self, 1, 0, dist, step_x.y)
            else:
                next_step = step_y.inspect(ray_info, self, 0, 1, dist, step_y.x)
            ray.append(next_step)
            origin = next_step
        return ray

    def update(self, dt):
        # can change lighting or modify the map here
        pass


class Point(object):
    def __init__(self, point, length=None):
        self.x = point[0]
        self.y = point[1]
        self.height = 0
        self.distance = 0
        self.length = length

    def step(self, rise, run, invert=False):
        if abs(run) > 1E-20:
            x, y = (self.y, self.x) if invert else (self.x, self.y)
            # this dx, dy math is critical!!!
            # doing it wrong will cause weird errors...
            dx = math.floor(x + 1) - x if run > 0 else math.ceil(x - 1) - x
            dy = dx * (rise / float(run))
            next_x = y + dy if invert else x + dx
            next_y = x + dx if invert else y + dy
            length = math.hypot(dx, dy)
        else:
            # Avoid / 0 errors
            next_x = next_y = None
            length = float("inf")
        return Point((next_x, next_y), length)

    def inspect(self, ray_info, game_map, shift_x, shift_y, distance, offset):
        # set height, distance if we hit a wall
        dx = shift_x if ray_info.cos < 0 else 0
        dy = shift_y if ray_info.sin < 0 else 0
        self.height = game_map.get(self.x - dx, self.y - dy)
        self.distance = distance + self.length
        self.offset = offset - math.floor(offset)
        return self


class Camera(object):
    def __init__(self, screen, resolution):
        self.screen = screen
        self.width, self.height = self.screen.get_size()
        # raycasting resolution
        self.resolution = float(resolution)
        # draw distance
        self.rnge = 14
        self.field_of_view = FIELD_OF_VIEW
        self.spacing = self.width / float(resolution)

    def render(self, player, game_map):
        self.draw_sky(player.direction, game_map.sky_texture)
        self.draw_columns(player, game_map)

    def draw_sky(self, direction, sky):
        # This also acts as "blanking"
        left = -sky.width * direction / (2 * math.pi)
        self.screen.blit(sky.image, (left, 0))
        if left < sky.width - self.width:
            self.screen.blit(sky.image, (left + sky.width, 0))

    def draw_columns(self, player, game_map):
        for column in range(int(self.resolution)):
            angle = self.field_of_view * (column / self.resolution - 0.5)
            point = (player.x, player.y)
            ray = game_map.cast_ray(point, player.direction + angle, self.rnge)
            self.draw_column(column, ray, angle, game_map)

    def draw_column(self, column, ray, angle, game_map):
        left = int(math.floor(column * self.spacing))
        for ray_index in range(len(ray) - 1, -1, -1):
            step = ray[ray_index]
            if step.height > 0:
                texture = game_map.wall_texture
                width = int(math.ceil(self.spacing))
                texture_x = int(texture.width * step.offset)
                wall = self.project(step.height, angle, step.distance)
                image_location = pg.Rect(texture_x, 0, 1, texture.height)
                image_slice = texture.image.subsurface(image_location)
                scale_rect = pg.Rect(left, wall.top, width, wall.height)
                scaled = pg.transform.scale(image_slice, scale_rect.size)
                self.screen.blit(scaled, scale_rect)
                return

    def project(self, height, angle, distance):
        z = max(distance * math.cos(angle), 0.2)
        wall_height = self.height * height / float(z)
        bottom = self.height / float(2.) * (1 + 1 / float(z))
        return WallInfo(bottom - wall_height, int(wall_height))


class Control(object):
    def __init__(self):
        self.screen = pg.display.get_surface()
        self.clock = pg.time.Clock()
        self.fps = 60.0
        self.keys = pg.key.get_pressed()
        self.done = False
        self.player = Player(15.3, -1.2, math.pi * 0.3)
        self.game_map = Map(32)
        self.camera = Camera(self.screen, 300)

    def event_loop(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True
            elif event.type in (pg.KEYDOWN, pg.KEYUP):
                self.keys = pg.key.get_pressed()

    def update(self, dt):
        self.game_map.update(dt)
        self.player.update(self.keys, dt, self.game_map)

    def display_fps(self):
        caption = "FPS: {:.2f}".format(self.clock.get_fps())
        pg.display.set_caption(caption)

    def main_loop(self):
        dt = self.clock.tick(self.fps) / 1000.
        while not self.done:
            self.event_loop()
            self.update(dt)
            self.camera.render(self.player, self.game_map)
            dt = self.clock.tick(self.fps) / 1000.
            pg.display.update()
            self.display_fps()

def main():
    os.environ["SDL_VIDEO_CENTERED"] = "True"
    pg.init()
    pg.display.set_mode(SCREEN_SIZE)
    Control().main_loop()
    pg.quit()
    sys.exit()

if __name__ == "__main__":
    if not os.path.exists("sky_texture.bmp"):
        print("sky texture not found, generating...")
        perlin("sky_texture.bmp", gradient=True)
    if not os.path.exists("wall_texture.bmp"):
        print("wall texture not found, generating...")
        perlin("wall_texture.bmp")
    main()
