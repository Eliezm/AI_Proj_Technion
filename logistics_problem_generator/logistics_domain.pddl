(define (domain logistics-strips)
  (:requirements :strips)
  (:predicates
    (OBJ ?obj)
    (TRUCK ?truck)
    (AIRPLANE ?airplane)
    (LOCATION ?loc)
    (CITY ?city)
    (AIRPORT ?airport)
    (at ?obj ?loc)
    (in ?obj1 ?obj2)
    (in-city ?loc ?city)
  )

  (:action LOAD-TRUCK
    :parameters (?obj ?truck ?loc)
    :precondition
      (and (OBJ ?obj) (TRUCK ?truck) (LOCATION ?loc)
           (at ?truck ?loc) (at ?obj ?loc))
    :effect
      (and (not (at ?obj ?loc)) (in ?obj ?truck))
  )

  (:action UNLOAD-TRUCK
    :parameters (?obj ?truck ?loc)
    :precondition
      (and (OBJ ?obj) (TRUCK ?truck) (LOCATION ?loc)
           (at ?truck ?loc) (in ?obj ?truck))
    :effect
      (and (not (in ?obj ?truck)) (at ?obj ?loc))
  )

  (:action LOAD-AIRPLANE
    :parameters (?obj ?airplane ?loc)
    :precondition
      (and (OBJ ?obj) (AIRPLANE ?airplane) (AIRPORT ?loc)
           (at ?airplane ?loc) (at ?obj ?loc))
    :effect
      (and (not (at ?obj ?loc)) (in ?obj ?airplane))
  )

  (:action UNLOAD-AIRPLANE
    :parameters (?obj ?airplane ?loc)
    :precondition
      (and (OBJ ?obj) (AIRPLANE ?airplane) (AIRPORT ?loc)
           (at ?airplane ?loc) (in ?obj ?airplane))
    :effect
      (and (not (in ?obj ?airplane)) (at ?obj ?loc))
  )

  (:action DRIVE-TRUCK
    :parameters (?truck ?loc-from ?loc-to ?city)
    :precondition
      (and (TRUCK ?truck) (LOCATION ?loc-from) (LOCATION ?loc-to) (CITY ?city)
           (at ?truck ?loc-from)
           (in-city ?loc-from ?city)
           (in-city ?loc-to ?city))
    :effect
      (and (not (at ?truck ?loc-from)) (at ?truck ?loc-to))
  )

  (:action FLY-AIRPLANE
    :parameters (?airplane ?loc-from ?loc-to)
    :precondition
      (and (AIRPLANE ?airplane) (AIRPORT ?loc-from) (AIRPORT ?loc-to)
           (at ?airplane ?loc-from))
    :effect
      (and (not (at ?airplane ?loc-from)) (at ?airplane ?loc-to))
  )
)
