(define (problem blocksworld-large-13)
  (:domain blocksworld)
  (:objects b0 b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b1) (on-table b2) (on-table b4) (on-table b5) (on b0 b7) (on b10 b1) (on b3 b4) (on b6 b5) (on b7 b9) (on b8 b0) (on b9 b3) (clear b10) (clear b2) (clear b6) (clear b8) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on b1 b0) (on b10 b9) (on b2 b1) (on b3 b2) (on b4 b3) (on b5 b4) (on b6 b5) (on b7 b6) (on b8 b7) (on b9 b8) (clear b10) (arm-empty))
  )
)
