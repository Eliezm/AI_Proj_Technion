(define (problem blocksworld-large-56)
  (:domain blocksworld)
  (:objects b0 b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b1) (on-table b2) (on-table b6) (on-table b8) (on b0 b5) (on b10 b6) (on b3 b9) (on b4 b1) (on b5 b8) (on b7 b10) (on b9 b4) (clear b0) (clear b2) (clear b3) (clear b7) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on b1 b0) (on b10 b9) (on b2 b1) (on b3 b2) (on b4 b3) (on b5 b4) (on b6 b5) (on b7 b6) (on b8 b7) (on b9 b8) (clear b10) (arm-empty))
  )
)
