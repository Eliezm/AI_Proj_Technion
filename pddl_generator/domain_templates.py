# -*- coding: utf-8 -*-
"""
Fixed domain definitions for problem generation.
These domains are well-studied and scalable.
"""

# ============================================================================
# DOMAIN 1: BLOCKS WORLD
# ============================================================================
BLOCKS_WORLD_DOMAIN = """(define (domain blocks-world)
  (:requirements :strips :typing)
  (:types block)
  (:predicates
    (on ?x ?y - block)
    (ontable ?x - block)
    (clear ?x - block)
    (holding ?x - block)
    (arm-empty)
  )

  (:action pick-up
    :parameters (?x - block)
    :precondition (and (clear ?x) (ontable ?x) (arm-empty))
    :effect (and (not (ontable ?x)) (not (clear ?x)) (not (arm-empty)) (holding ?x))
  )

  (:action put-down
    :parameters (?x - block)
    :precondition (holding ?x)
    :effect (and (ontable ?x) (clear ?x) (arm-empty) (not (holding ?x)))
  )

  (:action stack
    :parameters (?x ?y - block)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (on ?x ?y) (clear ?x) (not (holding ?x)) (arm-empty) (not (clear ?y)))
  )

  (:action unstack
    :parameters (?x ?y - block)
    :precondition (and (on ?x ?y) (clear ?x) (arm-empty))
    :effect (and (holding ?x) (clear ?y) (not (on ?x ?y)) (not (arm-empty)) (not (clear ?x)))
  )
)
"""

# ============================================================================
# DOMAIN 2: LOGISTICS
# ============================================================================
LOGISTICS_DOMAIN = """(define (domain logistics)
  (:requirements :strips :typing)
  (:types 
    truck location object city
  )
  (:predicates
    (in ?obj - object ?truck - truck)
    (at ?truck - truck ?loc - location)
    (at-obj ?obj - object ?loc - location)
    (connected ?from ?to - location)
    (in-city ?loc - location ?city - city)
    (obj-at-city ?obj - object ?city - city)
    (truck-at-city ?truck - truck ?city - city)
  )

  (:action load
    :parameters (?obj - object ?truck - truck ?loc - location)
    :precondition (and (at-obj ?obj ?loc) (at ?truck ?loc))
    :effect (and (in ?obj ?truck) (not (at-obj ?obj ?loc)))
  )

  (:action unload
    :parameters (?obj - object ?truck - truck ?loc - location)
    :precondition (and (in ?obj ?truck) (at ?truck ?loc))
    :effect (and (not (in ?obj ?truck)) (at-obj ?obj ?loc))
  )

  (:action drive
    :parameters (?truck - truck ?from ?to - location)
    :precondition (and (at ?truck ?from) (connected ?from ?to))
    :effect (and (at ?truck ?to) (not (at ?truck ?from)))
  )
)
"""

# ============================================================================
# DOMAIN 3: GRIPPER
# ============================================================================
GRIPPER_DOMAIN = """(define (domain gripper)
  (:requirements :strips :typing)
  (:types room object gripper)
  (:predicates
    (at-robot ?room - room)
    (at ?obj - object ?room - room)
    (free ?gripper - gripper)
    (carry ?obj - object ?gripper - gripper)
    (connect ?from ?to - room)
  )

  (:action move
    :parameters (?from ?to - room)
    :precondition (and (at-robot ?from) (connect ?from ?to))
    :effect (and (at-robot ?to) (not (at-robot ?from)))
  )

  (:action pick
    :parameters (?obj - object ?room - room ?gripper - gripper)
    :precondition (and (at-robot ?room) (at ?obj ?room) (free ?gripper))
    :effect (and (carry ?obj ?gripper) (not (at ?obj ?room)) (not (free ?gripper)))
  )

  (:action drop
    :parameters (?obj - object ?room - room ?gripper - gripper)
    :precondition (and (at-robot ?room) (carry ?obj ?gripper))
    :effect (and (at ?obj ?room) (free ?gripper) (not (carry ?obj ?gripper)))
  )
)
"""

DOMAINS = {
    "blocks_world": BLOCKS_WORLD_DOMAIN,
    "logistics": LOGISTICS_DOMAIN,
    "gripper": GRIPPER_DOMAIN,
}