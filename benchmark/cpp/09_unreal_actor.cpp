#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "GameFramework/SpringArmComponent.h"
#include "Camera/CameraComponent.h"
#include "InputActionValue.h"
#include "EnhancedInputComponent.h"
#include "EnhancedInputSubsystems.h"
#include "InputActionValue.h"
#include "Math/Vector.h"
#include "Math/Rotator.h"
#include "Containers/Array.h"
#include "Engine/World.h"
#include "Kismet/GameplayStatics.h"
#include "PhysicsEngine/PhysicsConstraintComponent.h"

UENUM(BlueprintType)
enum class ECharacterState : uint8 {
	Idle = 0,
	Moving = 1,
	Jumping = 2,
	Falling = 3,
	Attacking = 4
};

USTRUCT(BlueprintType)
struct FCharacterStats {
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Stats")
	float Health = 100.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Stats")
	float MaxHealth = 100.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Stats")
	float Stamina = 100.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Stats")
	float MaxStamina = 100.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Stats")
	float AttackPower = 25.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Stats")
	float DefensePower = 10.0f;
};

UCLASS()
class MYPROJECT_API ACustomCharacter : public ACharacter {
	GENERATED_BODY()

public:
	ACustomCharacter();

	virtual void BeginPlay() override;
	virtual void Tick(float DeltaTime) override;
	virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

	UFUNCTION(BlueprintCallable, Category = "Combat")
	void PerformAttack();

	UFUNCTION(BlueprintCallable, Category = "Combat")
	void TakeDamage(float DamageAmount);

	UFUNCTION(BlueprintCallable, Category = "Movement")
	void Jump() override;

	UFUNCTION(BlueprintCallable, Category = "State")
	void SetCharacterState(ECharacterState NewState);

	UFUNCTION(BlueprintPure, Category = "State")
	ECharacterState GetCharacterState() const { return CurrentState; }

	UFUNCTION(BlueprintPure, Category = "Stats")
	FCharacterStats GetStats() const { return CharacterStats; }

protected:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Camera")
	class USpringArmComponent* CameraBoom;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Camera")
	class UCameraComponent* FollowCamera;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat")
	class UAnimMontage* AttackMontage;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat")
	class UParticleSystem* HitParticles;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Sound")
	class USoundBase* AttackSound;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Movement")
	float WalkSpeed = 600.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Movement")
	float SprintSpeed = 1200.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Movement")
	float JumpForce = 1000.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Stats")
	FCharacterStats CharacterStats;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Combat")
	bool bIsAttacking = false;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Combat")
	float AttackCooldown = 0.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Combat")
	float AttackCooldownDuration = 1.0f;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Replication")
	bool bIsReplicated = true;

	ECharacterState CurrentState;

private:
	void MoveForward(const FInputActionValue& Value);
	void MoveRight(const FInputActionValue& Value);
	void Look(const FInputActionValue& Value);
	void Sprint();
	void StopSprinting();
	void UpdateAttackCooldown(float DeltaTime);
	void ApplyDamageEffect(const FVector& HitLocation);
};

ACustomCharacter::ACustomCharacter() {
	PrimaryActorTick.TickInterval = 0.0f;

	bUseControllerRotationPitch = false;
	bUseControllerRotationYaw = false;
	bUseControllerRotationRoll = false;

	GetCharacterMovement()->bOrientRotationToMovement = true;
	GetCharacterMovement()->MaxWalkSpeed = WalkSpeed;
	GetCharacterMovement()->MaxAcceleration = 2400.0f;

	GetMesh()->SetRelativeLocationAndRotation(
		FVector(0.0f, 0.0f, -90.0f),
		FRotator(0.0f, -90.0f, 0.0f)
	);

	CameraBoom = CreateDefaultSubobject<USpringArmComponent>(TEXT("CameraBoom"));
	CameraBoom->SetupAttachment(RootComponent);
	CameraBoom->TargetArmLength = 400.0f;
	CameraBoom->bUsePawnControlRotation = true;

	FollowCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("FollowCamera"));
	FollowCamera->SetupAttachment(CameraBoom, USpringArmComponent::SocketName);
	FollowCamera->bUsePawnControlRotation = false;

	CharacterStats.Health = 100.0f;
	CharacterStats.MaxHealth = 100.0f;
	CharacterStats.Stamina = 100.0f;
	CharacterStats.MaxStamina = 100.0f;
	CharacterStats.AttackPower = 25.0f;
	CharacterStats.DefensePower = 10.0f;

	CurrentState = ECharacterState::Idle;
	bIsReplicated = true;
}

void ACustomCharacter::BeginPlay() {
	Super::BeginPlay();

	if (APlayerController* PlayerController = Cast<APlayerController>(GetController())) {
		if (UEnhancedInputLocalPlayerSubsystem* Subsystem =
			PlayerController->GetLocalPlayer()->GetSubsystem<UEnhancedInputLocalPlayerSubsystem>()) {
			Subsystem->AddMappingContext(nullptr, 0);
		}
	}
}

void ACustomCharacter::Tick(float DeltaTime) {
	Super::Tick(DeltaTime);

	UpdateAttackCooldown(DeltaTime);

	if (GetCharacterMovement()->Velocity.Length() > 0.0f) {
		if (CurrentState != ECharacterState::Jumping && CurrentState != ECharacterState::Falling) {
			SetCharacterState(ECharacterState::Moving);
		}
	} else {
		if (CurrentState == ECharacterState::Moving) {
			SetCharacterState(ECharacterState::Idle);
		}
	}

	if (!GetCharacterMovement()->IsMovingOnGround()) {
		SetCharacterState(ECharacterState::Falling);
	}
}

void ACustomCharacter::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent) {
	Super::SetupPlayerInputComponent(PlayerInputComponent);

	if (UEnhancedInputComponent* EnhancedInputComponent =
		Cast<UEnhancedInputComponent>(PlayerInputComponent)) {
		EnhancedInputComponent->BindAction(nullptr, ETriggerEvent::Triggered, this, &ACustomCharacter::MoveForward);
		EnhancedInputComponent->BindAction(nullptr, ETriggerEvent::Triggered, this, &ACustomCharacter::MoveRight);
		EnhancedInputComponent->BindAction(nullptr, ETriggerEvent::Triggered, this, &ACustomCharacter::Look);
	}
}

void ACustomCharacter::MoveForward(const FInputActionValue& Value) {
	if (const float DirectionValue = Value.Get<float>()) {
		AddMovementInput(GetActorForwardVector(), DirectionValue);
	}
}

void ACustomCharacter::MoveRight(const FInputActionValue& Value) {
	if (const float DirectionValue = Value.Get<float>()) {
		AddMovementInput(GetActorRightVector(), DirectionValue);
	}
}

void ACustomCharacter::Look(const FInputActionValue& Value) {
	if (const FVector2D LookAxisValue = Value.Get<FVector2D>()) {
		AddControllerYawInput(LookAxisValue.X);
		AddControllerPitchInput(LookAxisValue.Y);
	}
}

void ACustomCharacter::Jump() {
	Super::Jump();
	SetCharacterState(ECharacterState::Jumping);
}

void ACustomCharacter::Sprint() {
	GetCharacterMovement()->MaxWalkSpeed = SprintSpeed;
}

void ACustomCharacter::StopSprinting() {
	GetCharacterMovement()->MaxWalkSpeed = WalkSpeed;
}

void ACustomCharacter::PerformAttack() {
	if (bIsAttacking || AttackCooldown > 0.0f) {
		return;
	}

	bIsAttacking = true;
	SetCharacterState(ECharacterState::Attacking);
	AttackCooldown = AttackCooldownDuration;

	if (AttackMontage) {
		PlayAnimMontage(AttackMontage);
	}

	FVector AttackLocation = GetActorLocation() + GetActorForwardVector() * 100.0f;
	ApplyDamageEffect(AttackLocation);

	bIsAttacking = false;
}

void ACustomCharacter::TakeDamage(float DamageAmount) {
	float ActualDamage = FMath::Max(0.0f, DamageAmount - CharacterStats.DefensePower);
	CharacterStats.Health -= ActualDamage;
	CharacterStats.Health = FMath::Clamp(CharacterStats.Health, 0.0f, CharacterStats.MaxHealth);

	if (CharacterStats.Health <= 0.0f) {
		Destroy();
	}
}

void ACustomCharacter::SetCharacterState(ECharacterState NewState) {
	if (CurrentState != NewState) {
		CurrentState = NewState;
	}
}

void ACustomCharacter::UpdateAttackCooldown(float DeltaTime) {
	if (AttackCooldown > 0.0f) {
		AttackCooldown -= DeltaTime;
	}
}

void ACustomCharacter::ApplyDamageEffect(const FVector& HitLocation) {
	if (HitParticles) {
		UGameplayStatics::SpawnEmitterAtLocation(
			GetWorld(),
			HitParticles,
			HitLocation,
			FRotator::ZeroRotator,
			FVector(1.0f, 1.0f, 1.0f)
		);
	}

	if (AttackSound) {
		UGameplayStatics::PlaySoundAtLocation(
			GetWorld(),
			AttackSound,
			HitLocation
		);
	}
}
