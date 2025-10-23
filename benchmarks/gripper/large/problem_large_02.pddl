(define (problem gripper-2)
  (:domain gripper)
  (:objects
    room0 room1 room2 room3 room4 - room
    obj0 obj1 obj2 obj3 obj4 obj5 obj6 obj7 obj8 obj9 obj10 obj11 - object
    gripper0 gripper1 - gripper
  )
  (:init
    (at-robot room0) (at obj0 room0) (at obj1 room0) (at obj2 room0) (at obj3 room0) (at obj4 room0) (at obj5 room0) (at obj6 room0) (at obj7 room0) (at obj8 room0) (at obj9 room0) (at obj10 room0) (at obj11 room0) (free gripper0) (free gripper1) (connect room0 room1) (connect room1 room0) (connect room1 room2) (connect room2 room1) (connect room2 room3) (connect room3 room2) (connect room3 room4) (connect room4 room3)
  )
  (:goal (and
    (at obj0 room4) (at obj1 room4) (at obj2 room4) (at obj3 room4) (at obj4 room4) (at obj5 room4) (at obj6 room4) (at obj7 room4) (at obj8 room4) (at obj9 room4) (at obj10 room4) (at obj11 room4)
  ))
)
