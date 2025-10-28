(define (problem blocksworld-large-83)
  (:domain blocksworld)
  (:objects b0 b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b1) (on-table b2) (on-table b8) (on b0 b3) (on b10 b5) (on b3 b9) (on b4 b2) (on b5 b1) (on b6 b4) (on b7 b8) (on b9 b10) (clear b0) (clear b6) (clear b7) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on-table b1) (on b10 b8) (on b2 b0) (on b3 b1) (on b4 b2) (on b5 b3) (on b6 b4) (on b7 b5) (on b8 b6) (on b9 b7) (clear b10) (clear b9) (arm-empty))
  )
)
