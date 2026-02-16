"""Flame steak - velocity field with XYZ-only evolution (no scale/rotation)"""
_base_ = './base_velocity.py'

ModelHiddenParams = dict(
    no_ds = True,  # No scale velocity
    no_dr = True,  # No rotation velocity
)
