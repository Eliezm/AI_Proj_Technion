(define (problem blocksworld-large-125)
  (:domain blocksworld)
  (:objects b0 b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b4) (on-table b6) (on-table b8) (on b0 b8) (on b1 b5) (on b10 b1) (on b2 b9) (on b3 b10) (on b5 b7) (on b7 b6) (on b9 b0) (clear b2) (clear b3) (clear b4) (arm-empty)
  )
  (:goal
    (and (on-table b10) (on-table b6) (on-table b7) (on b0 b4) (on b1 b9) (on b2 b1) (on b3 b10) (on b4 b2) (on b5 b0) (on b8 b7) (on b9 b8) (clear b3) (clear b5) (clear b6) (arm-empty))
  )
)
