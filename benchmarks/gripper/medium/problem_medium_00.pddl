(define (problem gripper-0)
  (:domain gripper)
  (:objects
    room0 room1 room2 room3 - room
    obj0 obj1 obj2 obj3 obj4 obj5 obj6 obj7 - object
    gripper0 gripper1 - gripper
  )
  (:init
    (at-robot room0) (at obj0 room0) (at obj1 room0) (at obj2 room0) (at obj3 room0) (at obj4 room0) (at obj5 room0) (at obj6 room0) (at obj7 room0) (free gripper0) (free gripper1) (connect room0 room1) (connect room1 room0) (connect room1 room2) (connect room2 room1) (connect room2 room3) (connect room3 room2)
  )
  (:goal (and
    (at obj0 room3) (at obj1 room3) (at obj2 room3) (at obj3 room3) (at obj4 room3) (at obj5 room3) (at obj6 room3) (at obj7 room3)
  ))
)
