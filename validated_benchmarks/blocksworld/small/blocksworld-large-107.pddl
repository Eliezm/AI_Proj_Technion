(define (problem blocksworld-large-107)
  (:domain blocksworld)
  (:objects b0 b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b0) (on-table b3) (on-table b8) (on-table b9) (on b1 b2) (on b10 b4) (on b2 b8) (on b4 b0) (on b5 b3) (on b6 b1) (on b7 b5) (clear b10) (clear b6) (clear b7) (clear b9) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on-table b1) (on b10 b8) (on b2 b0) (on b3 b1) (on b4 b2) (on b5 b3) (on b6 b4) (on b7 b5) (on b8 b6) (on b9 b7) (clear b10) (clear b9) (arm-empty))
  )
)
