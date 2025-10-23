(define (problem gripper-1)
  (:domain gripper)
  (:objects
    room0 room1 room2 - room
    obj0 obj1 obj2 obj3 - object
    gripper0 gripper1 - gripper
  )
  (:init
    (at-robot room0) (at obj0 room0) (at obj1 room0) (at obj2 room0) (at obj3 room0) (free gripper0) (free gripper1) (connect room0 room1) (connect room1 room0) (connect room1 room2) (connect room2 room1)
  )
  (:goal (and
    (at obj0 room2) (at obj1 room2) (at obj2 room2) (at obj3 room2)
  ))
)
