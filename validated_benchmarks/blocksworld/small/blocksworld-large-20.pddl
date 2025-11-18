(define (problem blocksworld-large-20)
  (:domain blocksworld)
  (:objects b0 b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b0) (on-table b10) (on-table b6) (on-table b8) (on b1 b9) (on b2 b7) (on b3 b0) (on b4 b10) (on b5 b3) (on b7 b5) (on b9 b2) (clear b1) (clear b4) (clear b6) (clear b8) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on b1 b0) (on b10 b9) (on b2 b1) (on b3 b2) (on b4 b3) (on b5 b4) (on b6 b5) (on b7 b6) (on b8 b7) (on b9 b8) (clear b10) (arm-empty))
  )
)
