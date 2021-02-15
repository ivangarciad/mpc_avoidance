#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to INVETT MPC controller.

"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    #sys.path.append(glob.glob('/home/ivan/Carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    #sys.path.append(glob.glob('/home/adolfo/Carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    sys.path.append(glob.glob('/home/ivan/Carla0.9.9.2/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    #sys.path.append(glob.glob('/home/ivan/Carla0.9.9.11/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import time
import sysv_ipc as ipc
import struct
from errno import ENOTCONN, EDEADLK, EAGAIN, EWOULDBLOCK
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
#sys.path.insert(0, '/home/mpc_path_planning')
import utils
import model_carla as model

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_g
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_l
    from pygame.locals import K_i
    from pygame.locals import K_z
    from pygame.locals import K_x
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
global trayectory_types
global trayectory_to_use
global use_road_speed

use_road_speed = False
input_is_right = False
trayectory_types = ['mix' ,'wide_turn', '90_degrees_turn', 'straight_line', 'turns_with_ramps', 'start_on_ramp']
trayectory_to_use = 'wide_turn'     #default trayectory in case it hasn't been specified on commands line

if len(sys.argv) > 2:
    for i in range(0,len(sys.argv)):
        if sys.argv[i] == '--path_type':
            if i+1 > len(sys.argv)-1:
                raise RuntimeError("Expected one argument for: --path_type")
            trayectory_to_use = sys.argv[i+1]
            for u in range(len(trayectory_types)):
                if trayectory_types[u] == trayectory_to_use:
                    input_is_right = True
            if input_is_right == True:
                pass
            else:
                raise RuntimeError('The path type selected is not valid. Use -h to view all path options.')
        if sys.argv[i] == '--use_road_speed':
            use_road_speed = True


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.player_obstacle_0 = None
        self.player_obstacle_1 = None
        self.player_obstacle_2 = None
        self.player_obstacle_3 = None
        self.player_obstacle_4 = None
        self.player_obstacle_5 = None

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter('vehicle.toyota.prius'))       #always spawn with Toyota Prius
        #blueprint_obstacle = random.choice(self.world.get_blueprint_library().filter('vehicle.audi.etron'))       #always spawn with Toyota Prius
        blueprint_obstacle = random.choice(self.world.get_blueprint_library().filter('walker.*'))       #Walker
        
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            color = '28,46,58'
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = blueprint.get_attribute('driver_id').recommended_values[10]
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
        else:
            print("No recommended values for 'speed' attribute")
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            if trayectory_to_use == trayectory_types[0]:      #Mix
                spawn_point = spawn_points[13] if spawn_points else carla.Transform()
            elif trayectory_to_use == trayectory_types[1]:    #Wide Turn
                spawn_point = spawn_points[30] if spawn_points else carla.Transform()
            elif trayectory_to_use == trayectory_types[2]:    #90 Degrees Turn
                spawn_point = spawn_points[33] if spawn_points else carla.Transform()
            elif trayectory_to_use == trayectory_types[3]:    #Straight Line
                spawn_point = spawn_points[60] if spawn_points else carla.Transform()
            elif trayectory_to_use == trayectory_types[4]:    #Turns with ramp
                spawn_point = spawn_points[51] if spawn_points else carla.Transform()
            elif trayectory_to_use == trayectory_types[5]:    #Start on ramp
                spawn_point = spawn_points[53] if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

            obstacle_flag = False
            if obstacle_flag == True:
              obstacle_place = carla.Transform(location = carla.Location(x=-88, y=90, z=spawn_point.location.z), rotation = spawn_point.rotation)
              self.player_obstacle_0 = self.world.try_spawn_actor(blueprint_obstacle, obstacle_place)
              obstacle_place = carla.Transform(location = carla.Location(x=-88, y=165, z=spawn_point.location.z), rotation = carla.Rotation(pitch=0, roll=0, yaw=-90))
              self.player_obstacle_1 = self.world.try_spawn_actor(blueprint_obstacle, obstacle_place)
              obstacle_place = carla.Transform(location = carla.Location(x=-9, y=207.3, z=spawn_point.location.z), rotation = carla.Rotation(pitch=0, roll=0, yaw=-75))
              self.player_obstacle_2 = self.world.try_spawn_actor(blueprint_obstacle, obstacle_place)
              obstacle_place = carla.Transform(location = carla.Location(x=60.2, y=207.4, z=spawn_point.location.z), rotation = carla.Rotation(pitch=0, roll=0, yaw=0))
              self.player_obstacle_3 = self.world.try_spawn_actor(blueprint_obstacle, obstacle_place)
              obstacle_place = carla.Transform(location = carla.Location(x=222.7, y=182.6, z=spawn_point.location.z), rotation = carla.Rotation(pitch=0, roll=0, yaw=-42))
              self.player_obstacle_4 = self.world.try_spawn_actor(blueprint_obstacle, obstacle_place)
              obstacle_place = carla.Transform(location = carla.Location(x=243.7, y=80, z=spawn_point.location.z), rotation = carla.Rotation(pitch=0, roll=0, yaw=-89))
              self.player_obstacle_5 = self.world.try_spawn_actor(blueprint_obstacle, obstacle_place)

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor,
            self.player,
            self.player_obstacle_0,
            self.player_obstacle_1,
            self.player_obstacle_2,
            self.player_obstacle_3,
            self.player_obstacle_4,
            self.player_obstacle_5]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

       #if not self._autopilot_enabled:
       #    if isinstance(self._control, carla.VehicleControl):
       #        self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
       #        self._control.reverse = self._control.gear < 0
       #    elif isinstance(self._control, carla.WalkerControl):
       #        self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
       #    world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()

        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        for actor in world.world.get_actors():
            if actor.attributes.get('role_name') == 'hero':
                player = actor
                break
        self._info_text = [
            'MPC INVETT',
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            'Speed:   % 15.0f m/s' % (math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            'Yaw: % 16.0f degrees' % t.rotation.yaw,
            'Driver-ID: % 18.0f' % player.id,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], sensor_data.gyroscope.x)),
            max(limits[0], min(limits[1], sensor_data.gyroscope.y)),
            max(limits[0], min(limits[1], sensor_data.gyroscope.z)))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=2.8, z=1.0),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        # Camera position
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-4.0, z=5.0), carla.Rotation(pitch=30.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}]]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '50')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype = int)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    flag_veh_control = True
    flag_save_results = True
    flag_straight_path = False
    flag_print_results_straight = False
    flag_print_results_roundabout = False
    flag_print_results_90turn = False
    wheelbase_para = 1.6
    i=1
    
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(6.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world, args.autopilot)
        
        print (world.player)
        print (world.player_obstacle_0)
        print (world.player_obstacle_1)
        print (world.player_obstacle_2)
        print (world.player_obstacle_3)
        print (world.player_obstacle_4)
        print (world.player_obstacle_5)

        if isinstance(world.player, carla.Vehicle):
            control = carla.VehicleControl()

        if isinstance(world.player_obstacle_0, carla.Vehicle):
            control_obstacle = carla.VehicleControl()

        clock = pygame.time.Clock()
        
        waypoints = world.map.get_waypoint(world.player.get_location(), project_to_road=False, lane_type=(carla.LaneType.Driving))
        waypoint_list = [waypoints]
        x_ref, y_ref, yaw_ref = [], [], []

        #Define a path using waypoints separated 0.5 meters from each other
        for i in range(0,7000):
            next_waypoint = waypoints.next(0.5)
            if flag_straight_path == False:
                dir_num = len(next_waypoint)
                if dir_num == 1:
                        waypoint_list.append(next_waypoint[-1])
                elif dir_num >= 2:
                    if i > 200 and i < 300:
                        waypoint_list.append(next_waypoint[-1])
                    else:
                        waypoint_list.append(next_waypoint[0])
            else:
                waypoint_list.append(next_waypoint[-1])
            waypoints = waypoint_list[-1]

        waypoints_new = [waypoint_list[0]]
        for elem in waypoint_list:
          if ((elem.transform.location.x - waypoints_new[-1].transform.location.x)**2 + (elem.transform.location.y - waypoints_new[-1].transform.location.y)**2) > 0.4**2:
              waypoints_new.append(elem)

        if use_road_speed == True:
            v_ref = (world.player.get_speed_limit()) #Speed reference m/s
            print (v_ref)
        else:
            v_ref = 5 

        #Build the reference state vector and reference state file
        reference_file = open('reference.json', 'w')

        # Draw the reference with Blue points
        for waypoint in waypoints_new:
            #world.world.debug.draw_point(carla.Location(x=waypoint.transform.location.x, y=waypoint.transform.location.y, z=waypoint.transform.location.z + 2.), size=0.05, color=carla.Color(0,0,255), life_time=1200, persistent_lines=False)
            x_ref.append(waypoint.transform.location.x)
            y_ref.append(waypoint.transform.location.y)
            yaw_ref.append(np.deg2rad(waypoint.transform.rotation.yaw))
            data = {"x_ref": x_ref[-1], "y_ref": y_ref[-1], "yaw_ref": yaw_ref[-1], "v_ref": v_ref}
            json.dump(data, reference_file)
            reference_file.write('\n')

        reference_file.close()

        # Draw obstacle
        #obstacle_place = carla.Location(x=-87, y=106, z=waypoint.transform.location.z + 2.)
        #world.world.debug.draw_point(obstacle_place, size=0.1, color=carla.Color(0,0,255), life_time=1200, persistent_lines=False)
        
        i = 1
        print ('The way points have been loaded')
        collision_sensor_start = world.collision_sensor.get_collision_history()
        collision_flag = 0.0

        # MPC parameters
        N = 6
        sol_mpc = np.asarray(np.zeros(12))
        max_steer = np.deg2rad(60)
        max_acc_veh = 2 #m/s^2
        min_acc_veh = 3 #m/s^2

        offset = 10

        # Shared memory set up
        path = "/tmp"
        key_send = ipc.ftok(path, 2333)
        key_recv = ipc.ftok(path, 2444)
        key_recv_pol = ipc.ftok(path, 2555)
        key_recv_tentatives = ipc.ftok(path, 2666)
        key_recv_tentatives_heading = ipc.ftok(path, 2777)

        shm_send = ipc.SharedMemory(key_send, flags=ipc.IPC_CREAT, size=9*8)
        shm_recv = ipc.SharedMemory(key_recv, flags=ipc.IPC_CREAT, size=13*8)
        shm_recv_pol = ipc.SharedMemory(key_recv_pol, flags=ipc.IPC_CREAT, size=16*8)
        shm_recv_tentatives = ipc.SharedMemory(key_recv_tentatives, flags=ipc.IPC_CREAT, size=12*8)
        shm_recv_tentatives_heading = ipc.SharedMemory(key_recv_tentatives_heading, flags=ipc.IPC_CREAT, size=6*8)

        shm_send.attach(0,0)
        shm_recv.attach(0,0)
        shm_recv_pol.attach(0,0)
        shm_recv_tentatives.attach(0,0)
        shm_recv_tentatives_heading.attach(0,0)

        #control_obstacle.throttle = 0.2 
        #control_obstacle.brake = 0 
        #world.player_obstacle.apply_control(control_obstacle)
        
        m = 1775
        rw = 0.37
        xd = 1
        eficiency = 1

        #Game loop
        while True:
            start_time = time.time()
            clock.tick_busy_loop(60)
            controller.parse_events(client, world, clock)
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            if flag_veh_control == True:

               #c = world.player.get_control()
               #p = world.player.get_physics_control()
               #engine_rpm = p.max_rpm * c.throttle
               #if c.gear > 0:
               #    gear = p.forward_gears[c.gear-1]
               #    engine_rpm *= gear.ratio
               #    print ('RPM: ' + str(engine_rpm))
               #    ax = 400*c.throttle*gear.ratio*xd*eficiency/(m*rw)
               #    print ('ax: ' + str(ax))

                #Detect collision and reset position
                collision_sensor_now = world.collision_sensor.get_collision_history()
                if collision_sensor_now != collision_sensor_start:
                    collision_flag = 1.0

                #Collect data from Vehicle
    
                accv_y = world.imu_sensor.accelerometer[1]
                gyro_z = -world.imu_sensor.gyroscope[2]
                t = world.player.get_transform()
                v = world.player.get_velocity()
                acc = world.player.get_acceleration()
                x_act = round(t.location.x,3)
                y_act = round(t.location.y,3)
                yaw_act = round(np.deg2rad(t.rotation.yaw), 3)
                speed_veh = round(math.sqrt(v.x**2 + v.y**2), 3)

                x_state_vector = [x_act, y_act, yaw_act, speed_veh, -control.steer*max_steer, accv_y, gyro_z, collision_flag]
                #print (x_state_vector)
                
                #Send data to Gateway_mpc.
                data_send = struct.pack('@ddddddddd', x_act, y_act, yaw_act, speed_veh, -control.steer*max_steer, accv_y, gyro_z, control.throttle, collision_flag)
                #data_send = struct.pack('@dddddddd', x_act, y_act, yaw_act, speed_veh, control.steer*max_steer, 0, 0, collision_flag)
                shm_send.write(data_send)

                #Receive data from Gateway_mpc.
                buf_recv = shm_recv.read(13*8)
                sol_mpc = struct.unpack('@ddddddddddddd', buf_recv)

                buf_recv_pol = shm_recv_pol.read(16*8)
                points_poly = struct.unpack('@dddddddddddddddd', buf_recv_pol)

                buf_recv_tentatives = shm_recv_tentatives.read(12*8)
                tentatives = struct.unpack('@dddddddddddd', buf_recv_tentatives)

                buf_recv_tentatives_heading = shm_recv_tentatives_heading.read(6*8)
                tentatives_heading = struct.unpack('@dddddd', buf_recv_tentatives_heading)
                
                # Car position with tentatives (Green color).
                for index, index_heading in zip(range(0,12,2), range(0,6)):
                    begin_point = carla.Location(x=tentatives[index], y=tentatives[index+1], z=t.location.z + 2.15)
                    end_point = carla.Location(x=tentatives[index] + 0.5*np.cos(tentatives_heading[index_heading]), y=tentatives[index+1] + 0.5*np.sin(tentatives_heading[index_heading]), z=t.location.z + 2.15)
                    world.world.debug.draw_arrow(begin_point, end_point, thickness=0.03, arrow_size=0.03, color=carla.Color(0,255,0), life_time=0.05)

                # Poinamial points (White color).
                for index in range(0,16, 2):
                    location = carla.Location(x=points_poly[index], y=points_poly[index+1], z=t.location.z + 2.02)
                    world.world.debug.draw_point(location, size=0.05, color=carla.Color(255,255,255), life_time=0.05, persistent_lines=False)

                # Reference points having in account (Red color).
                current_index = round(sol_mpc[12])
                for u in range(current_index, current_index+N+offset):
                    location = carla.Location(x=x_ref[u], y=y_ref[u], z=t.location.z + 2.1)
                    world.world.debug.draw_point(location, size=0.05, color=carla.Color(255,0,0), life_time=0.1, persistent_lines=False)
                
                # Appling Control Actions.
                # A scalar value to control the vehicle steering [-1.0, 1.0]
                offset_controller = 0
                #control.steer = model.torque_to_steer(sol_mpc[N+offset_controller], control.steer*max_steer)/max_steer
                #control.steer = sol_mpc[N+offset_controller]/max_steer
                control.steer = -sol_mpc[0+offset_controller]/max_steer

                if sol_mpc[N+offset_controller] >= 0:
                    # A scalar value to control the vehicle throttle [0.0, 1.0]
                    #control.throttle = sol_mpc[0+offset_controller]/max_acc_veh 
                    if sol_mpc[N+offset_controller] > 0.8:
                        control.throttle = 0.8
                    else:
                        control.throttle = sol_mpc[N+offset_controller]

                    control.brake = 0 
                else:
                    # A scalar value to control the vehicle brake [0.0, 1.0]
                    control.brake = abs(sol_mpc[N+offset_controller])/2
                    #control.brake = 0 
                    control.throttle = 0 

                # The simulator is restarted.
                if collision_sensor_now != collision_sensor_start:
                    print("Collision detected!!!")
                    if (world and world.recording_enabled):
                        client.stop_recorder()
                    if world is not None:
                        world.destroy()
                    hud = HUD(args.width, args.height)
                    world = World(client.get_world(), hud, args)
                    controller = KeyboardControl(world, args.autopilot)
                    collision_flag = 0.0
                    current_index = 0
                    control.steer = 0
                    control.throttle = 0 
                    control.brake = 0 

                world.player.apply_control(control)
                
                

    finally:
        
        shm_send.detach()
        shm_recv.detach()
        shm_recv_pol.detach()
        shm_recv_tentatives.detach()
        shm_recv_tentatives_heading.detach()

        shm_send.remove()
        shm_recv.remove()
        shm_recv_pol.remove()
        shm_recv_tentatives.remove()
        shm_recv_tentatives_heading.remove()

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--path_type',
        action='store',
        dest='path_type',
        metavar='NAME',
        default='mix',
        help='Options are: mix , wide_turn, 90_degrees_turn, straight_line, turns_with_ramps, start_on_ramp')
    argparser.add_argument(
        '--obstacles',
        default=False,
        type=bool,
        help='Obstacles in the world (default: False)')
    argparser.add_argument(
        '--use_road_speed',
        action='store_true',
        help='Use the road speed limit as the vehicle target speed')
    args = argparser.parse_args()
    trayectory_to_use = args.path_type
    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
