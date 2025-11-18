(define (problem blocksworld-large-188)
  (:domain blocksworld)
  (:objects b0 b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b3) (on-table b4) (on-table b7) (on b0 b3) (on b1 b2) (on b10 b6) (on b2 b9) (on b5 b1) (on b6 b5) (on b8 b10) (on b9 b0) (clear b4) (clear b7) (clear b8) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on b1 b0) (on b10 b9) (on b2 b1) (on b3 b2) (on b4 b3) (on b5 b4) (on b6 b5) (on b7 b6) (on b8 b7) (on b9 b8) (clear b10) (arm-empty))
  )
)
