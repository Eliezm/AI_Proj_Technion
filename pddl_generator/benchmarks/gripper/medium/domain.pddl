(define (domain gripper)
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
