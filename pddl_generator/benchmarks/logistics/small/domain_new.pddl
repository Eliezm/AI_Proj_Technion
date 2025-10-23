(define (domain logistics)
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
